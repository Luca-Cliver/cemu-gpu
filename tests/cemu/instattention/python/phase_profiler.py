import csv
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, get_native_id
from typing import Optional


_request_context = ContextVar("cemu_profile_request", default=None)


def request_context(profiler):
    return _request_context.get() if profiler is not None and profiler.interval_path else None


@contextmanager
def profile_request(profiler, context):
    if profiler is None or not profiler.interval_path:
        yield
        return
    token = _request_context.set(context)
    try:
        yield
    finally:
        _request_context.reset(token)


@contextmanager
def profile_lock(profiler, name, lock):
    detailed = stack_profiler(profiler)
    if detailed is None:
        with lock:
            yield
        return
    acquired = False
    try:
        with profile_scope(detailed, name + ".lock_wait_wall"):
            lock.acquire()
            acquired = True
        yield
    finally:
        if acquired:
            lock.release()


@dataclass(frozen=True)
class PhaseProfile:
    name: str
    count: int
    total_ns: int
    maximum_ns: int
    byte_count: int

    @property
    def average_ns(self) -> float:
        return self.total_ns / self.count if self.count else 0.0

    @property
    def gib_per_second(self) -> float:
        if self.byte_count == 0 or self.total_ns == 0:
            return 0.0
        return (self.byte_count / 1024**3) / (self.total_ns / 1e9)


class PhaseProfiler:
    """Aggregate phase timing with opt-in buffered request intervals."""

    def __init__(self, stack=False):
        self.stack = stack
        self._lock = Lock()
        self._metrics = {}
        self._metadata = {}
        self._cuda_events = deque()
        self._cuda_lock = Lock()
        self.interval_path = os.environ.get("CEMU_PROFILE_INTERVALS")
        self._intervals = []
        self._interval_limit = int(os.environ.get("CEMU_PROFILE_INTERVAL_LIMIT", "500000")) if self.interval_path else 0
        self._intervals_dropped = 0
        if self.interval_path and self._interval_limit < 1:
            raise ValueError("CEMU_PROFILE_INTERVAL_LIMIT must be positive")

    def defer_cuda(self, name, start, end):
        # Harvest completed events without synchronizing the inference stream.
        with self._cuda_lock:
            self._cuda_events.append((name, start, end))
        self.collect_cuda()

    def collect_cuda(self, wait=False):
        with self._cuda_lock:
            while self._cuda_events:
                name, start, end = self._cuda_events[0]
                if wait:
                    end.synchronize()
                elif not end.query():
                    break
                self.record(name, round(start.elapsed_time(end) * 1_000_000))
                self._cuda_events.popleft()

    @contextmanager
    def measure(self, name: str, byte_count: int = 0):
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            self.record(name, end_ns - start_ns, byte_count)
            self.trace_interval(name, start_ns, end_ns)

    def trace_interval(self, name, start_ns, end_ns):
        if self.interval_path:
            if end_ns < start_ns:
                raise ValueError("profile interval ends before it starts")
            with self._lock:
                if len(self._intervals) < self._interval_limit:
                    self._intervals.append(dict(name=name, start_ns=start_ns, end_ns=end_ns,
                                                tid=get_native_id(), request=request_context(self)))
                else:
                    self._intervals_dropped += 1

    def write_intervals(self):
        if not self.interval_path:
            return
        with self._lock:
            intervals = tuple(self._intervals)
            dropped = self._intervals_dropped
        path = Path(self.interval_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clocksource = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
        report = dict(version=1, clock=time.get_clock_info("perf_counter").implementation,
                      clocksource=clocksource.read_text().strip() if clocksource.exists() else None,
                      stack=self.stack, dropped=dropped, request_context_version=1, intervals=intervals,
                      metadata=self.metadata_snapshot())
        path.write_text(json.dumps(report) + "\n")

    def record_metadata_once(self, name, value):
        with self._lock:
            self._metadata.setdefault(name, value)

    def metadata_snapshot(self):
        with self._lock:
            return dict(self._metadata)

    def record(self, name: str, elapsed_ns: int, byte_count: int = 0) -> None:
        if not name:
            raise ValueError("profile phase name is required")
        if elapsed_ns < 0 or byte_count < 0:
            raise ValueError("profile time and byte count must be non-negative")
        with self._lock:
            count, total_ns, maximum_ns, total_bytes = self._metrics.get(
                name,
                (0, 0, 0, 0),
            )
            self._metrics[name] = (
                count + 1,
                total_ns + elapsed_ns,
                max(maximum_ns, elapsed_ns),
                total_bytes + byte_count,
            )

    def snapshot(self):
        self.collect_cuda(wait=True)
        with self._lock:
            metrics = dict(self._metrics)
        return tuple(
            PhaseProfile(name, count, total_ns, maximum_ns, byte_count)
            for name, (count, total_ns, maximum_ns, byte_count) in sorted(
                metrics.items()
            )
        )

    def write_csv(self, path) -> Path:
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = (
                "phase",
                "count",
                "total_seconds",
                "average_us",
                "maximum_us",
                "bytes",
                "gib",
                "effective_gib_per_second",
            )
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.snapshot():
                writer.writerow(
                    {
                        "phase": metric.name,
                        "count": metric.count,
                        "total_seconds": f"{metric.total_ns / 1e9:.9f}",
                        "average_us": f"{metric.average_ns / 1e3:.3f}",
                        "maximum_us": f"{metric.maximum_ns / 1e3:.3f}",
                        "bytes": metric.byte_count,
                        "gib": f"{metric.byte_count / 1024**3:.6f}",
                        "effective_gib_per_second": f"{metric.gib_per_second:.6f}",
                    }
                )
        self.write_intervals()
        metadata = self.metadata_snapshot()
        if metadata:
            output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
                json.dumps(metadata) + "\n")
        return output_path

    def summary(self):
        lines = []
        for metric in sorted(
            self.snapshot(),
            key=lambda item: item.total_ns,
            reverse=True,
        ):
            line = (
                f"{metric.name}: total={metric.total_ns / 1e9:.6f}s, "
                f"count={metric.count}, avg={metric.average_ns / 1e3:.3f}us, "
                f"max={metric.maximum_ns / 1e3:.3f}us"
            )
            if metric.byte_count:
                line += (
                    f", data={metric.byte_count / 1024**3:.3f}GiB, "
                    f"throughput={metric.gib_per_second:.3f}GiB/s"
                )
            lines.append(line)
        return tuple(lines)


@contextmanager
def profile_scope(
    profiler: Optional[PhaseProfiler],
    name: str,
    byte_count: int = 0,
):
    if profiler is None:
        yield
        return
    with profiler.measure(name, byte_count):
        yield


def stack_profiler(profiler):
    return profiler if profiler is not None and profiler.stack else None


def submit_profiled(submit, function, profiler, name, *args):
    detailed = stack_profiler(profiler)
    if detailed is None:
        return submit(function, *args)
    submitted_ns = time.perf_counter_ns()
    context = copy_context()

    def run():
        started_ns = time.perf_counter_ns()
        detailed.record(name + ".queue_wait_wall", started_ns - submitted_ns)
        detailed.trace_interval(name + ".queue_wait_wall", submitted_ns, started_ns)
        with profile_scope(detailed, name + ".worker_wall"):
            return function(*args)

    return submit(context.run, run)


def profile_client_call(profiler, name, function, *args):
    """Client intervals include IO waiting, not just Guest software execution."""
    detailed = stack_profiler(profiler)
    if detailed is None:
        return function(*args)
    with profile_scope(detailed, name + ".call_wall"):
        result, timings = function(*args, profile=True)
    for field in ("binding_prepare_wall", "buffer_prepare_wall", "io_wait_wall",
                  "buffer_finish_wall", "binding_finish_wall"):
        detailed.record(name + "." + field, timings[field],
                        timings["io_bytes"] if field == "io_wait_wall" else 0)
    return result


@contextmanager
def profile_tensor_scope(profiler, name, device):
    """Caller wall time and deferred CUDA stream time; no per-phase sync.

    CUDA intervals include stream stalls, not just kernel execution. Wall time
    can include blocking copies; the two metrics must not be added together.
    Use on the inference thread; call snapshot after workers have joined.
    """
    if profiler is None:
        yield
        return
    import torch
    device = torch.device(device)
    start = end = None
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start.record(stream)
    try:
        with profiler.measure(name + ".host_wall"):
            yield
    finally:
        if end is not None:
            end.record(stream)
            profiler.defer_cuda(name + ".cuda_stream", start, end)
