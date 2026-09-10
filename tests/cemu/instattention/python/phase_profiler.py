import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional


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
    """Thread-safe, aggregate-only phase timing for long pipeline runs."""

    def __init__(self):
        self._lock = Lock()
        self._metrics = {}

    @contextmanager
    def measure(self, name: str, byte_count: int = 0):
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(name, time.perf_counter_ns() - start_ns, byte_count)

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
