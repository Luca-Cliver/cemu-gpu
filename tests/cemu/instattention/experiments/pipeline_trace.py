import csv
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PipelineEvent:
    timestamp_ns: int
    thread_name: str
    message: str


@dataclass(frozen=True)
class PipelineInterval:
    phase: str
    operation: str
    identity: str
    layer: Optional[int]
    request_id: Optional[int]
    token: Optional[int]
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PipelineOverlap:
    phase: str
    first_operation: str
    second_operation: str
    first_identity: str
    second_identity: str
    first_start_ns: int
    first_end_ns: int
    second_start_ns: int
    second_end_ns: int
    overlap_ns: int


class PipelineTrace:
    low_overhead = True

    _PREFILL_PATTERN = re.compile(
        r"^\[prefill\] layer=(?P<layer>\d+) "
        r"(?P<operation>Attention|MLP)-(?P<state>start|complete)"
    )
    _PREFILL_STORE_PATTERN = re.compile(
        r"^\[prefill-kv-writer\] store-(?P<state>start|complete) "
        r"layer=(?P<layer>\d+)"
    )
    _DECODE_SCHEDULER_PATTERN = re.compile(
        r"^\[cemu-attention-slot-scheduler\] "
        r"(?P<operation>prefetch|load|compute)-(?P<state>start|complete) "
        r"request=(?P<request>\d+).*layer=(?P<layer>\d+)"
    )
    _DECODE_STORE_PATTERN = re.compile(
        r"^\[kv-backend\] store-(?P<state>start|complete) "
        r"request=(?P<request>\d+), layer=(?P<layer>\d+), token=(?P<token>\d+)"
    )
    _DECODE_QKV_PATTERN = re.compile(
        r"^\[decode\] QKV-(?P<state>start|complete) "
        r"position=(?P<token>\d+), layer=(?P<layer>\d+)"
        r"(?:, microbatch=(?P<microbatch>\d+))?"
    )

    def __init__(self):
        self._start_ns = time.perf_counter_ns()
        self._events: List[PipelineEvent] = []
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        self.record(message)

    def record(
        self,
        message: str,
        timestamp_ns: Optional[int] = None,
        thread_name: Optional[str] = None,
    ) -> None:
        event = PipelineEvent(
            timestamp_ns=(
                time.perf_counter_ns() if timestamp_ns is None else timestamp_ns
            ),
            thread_name=(
                threading.current_thread().name
                if thread_name is None
                else thread_name
            ),
            message=message,
        )
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> Tuple[PipelineEvent, ...]:
        with self._lock:
            return tuple(sorted(self._events, key=lambda event: event.timestamp_ns))

    def intervals(self) -> Tuple[PipelineInterval, ...]:
        starts: Dict[Tuple[str, str, str], List[PipelineEvent]] = {}
        intervals = []
        for event in self.events:
            marker = self._parse_marker(event.message)
            if marker is None:
                continue
            phase, operation, state, identity, layer, request_id, token = marker
            key = (phase, operation, identity)
            if state == "start":
                starts.setdefault(key, []).append(event)
                continue
            pending = starts.get(key)
            if not pending:
                continue
            start = pending.pop(0)
            intervals.append(
                PipelineInterval(
                    phase=phase,
                    operation=operation,
                    identity=identity,
                    layer=layer,
                    request_id=request_id,
                    token=token,
                    start_ns=start.timestamp_ns,
                    end_ns=event.timestamp_ns,
                )
            )
        return tuple(sorted(intervals, key=lambda interval: interval.start_ns))

    def overlaps(self) -> Tuple[PipelineOverlap, ...]:
        intervals = self.intervals()
        prefill = [item for item in intervals if item.phase == "prefill"]
        decode = [item for item in intervals if item.phase == "decode"]
        overlaps = []

        for store in self._select(prefill, "store"):
            for mlp in self._select(prefill, "mlp"):
                if store.layer == mlp.layer:
                    self._append_overlap(overlaps, store, mlp)
            for attention in self._select(prefill, "attention"):
                if store.layer is not None and attention.layer == store.layer + 1:
                    self._append_overlap(overlaps, store, attention)

        for compute in self._select(decode, "compute"):
            for store in self._select(decode, "store"):
                if compute.request_id == store.request_id:
                    self._append_overlap(overlaps, compute, store)
            for load in self._select(decode, "load"):
                if load.request_id != compute.request_id:
                    self._append_overlap(overlaps, compute, load)

        for qkv in self._select(decode, "qkv"):
            for store in self._select(decode, "store"):
                if qkv.token == store.token and qkv.layer != store.layer:
                    self._append_overlap(overlaps, qkv, store)

        return tuple(
            sorted(
                overlaps,
                key=lambda item: (
                    item.phase,
                    item.first_start_ns,
                    item.second_start_ns,
                ),
            )
        )

    def write(self, event_path: Path) -> Tuple[Path, Path]:
        event_path = Path(event_path)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        overlap_path = event_path.with_name(
            f"{event_path.stem}-overlap{event_path.suffix or '.csv'}"
        )
        events = self.events
        with event_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp_ns", "elapsed_us", "thread", "message"))
            for event in events:
                writer.writerow(
                    (
                        event.timestamp_ns,
                        f"{(event.timestamp_ns - self._start_ns) / 1000.0:.3f}",
                        event.thread_name,
                        event.message,
                    )
                )

        with overlap_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "phase",
                    "first_operation",
                    "second_operation",
                    "first_identity",
                    "second_identity",
                    "first_start_us",
                    "first_end_us",
                    "second_start_us",
                    "second_end_us",
                    "overlap_us",
                )
            )
            for overlap in self.overlaps():
                writer.writerow(
                    (
                        overlap.phase,
                        overlap.first_operation,
                        overlap.second_operation,
                        overlap.first_identity,
                        overlap.second_identity,
                        f"{(overlap.first_start_ns - self._start_ns) / 1000.0:.3f}",
                        f"{(overlap.first_end_ns - self._start_ns) / 1000.0:.3f}",
                        f"{(overlap.second_start_ns - self._start_ns) / 1000.0:.3f}",
                        f"{(overlap.second_end_ns - self._start_ns) / 1000.0:.3f}",
                        f"{overlap.overlap_ns / 1000.0:.3f}",
                    )
                )
        return event_path, overlap_path

    def summary(self) -> Tuple[str, ...]:
        overlaps = self.overlaps()
        groups: Dict[Tuple[str, str, str], List[int]] = {}
        for overlap in overlaps:
            key = (
                overlap.phase,
                overlap.first_operation,
                overlap.second_operation,
            )
            groups.setdefault(key, []).append(overlap.overlap_ns)
        lines = []
        for (phase, first, second), values in sorted(groups.items()):
            lines.append(
                f"{phase} {first}+{second}: count={len(values)}, "
                f"total={sum(values) / 1e6:.3f} ms, "
                f"mean={sum(values) / len(values) / 1e6:.3f} ms"
            )
        return tuple(lines)

    @classmethod
    def _parse_marker(cls, message: str):
        match = cls._PREFILL_PATTERN.match(message)
        if match:
            layer = int(match.group("layer"))
            operation = match.group("operation").lower()
            return (
                "prefill",
                operation,
                match.group("state"),
                f"layer={layer}",
                layer,
                None,
                None,
            )

        match = cls._PREFILL_STORE_PATTERN.match(message)
        if match:
            layer = int(match.group("layer"))
            return (
                "prefill",
                "store",
                match.group("state"),
                f"layer={layer}",
                layer,
                None,
                None,
            )

        match = cls._DECODE_SCHEDULER_PATTERN.match(message)
        if match:
            request_id = int(match.group("request"))
            layer = int(match.group("layer"))
            operation = match.group("operation")
            if operation == "prefetch":
                operation = "load"
            return (
                "decode",
                operation,
                match.group("state"),
                f"request={request_id},layer={layer}",
                layer,
                request_id,
                None,
            )

        match = cls._DECODE_STORE_PATTERN.match(message)
        if match:
            request_id = int(match.group("request"))
            layer = int(match.group("layer"))
            token = int(match.group("token"))
            return (
                "decode",
                "store",
                match.group("state"),
                f"request={request_id},layer={layer},token={token}",
                layer,
                request_id,
                token,
            )

        match = cls._DECODE_QKV_PATTERN.match(message)
        if match:
            layer = int(match.group("layer"))
            token = int(match.group("token"))
            microbatch = match.group("microbatch")
            identity = f"position={token},layer={layer}"
            if microbatch is not None:
                identity += f",microbatch={microbatch}"
            return (
                "decode",
                "qkv",
                match.group("state"),
                identity,
                layer,
                None,
                token,
            )
        return None

    @staticmethod
    def _select(
        intervals: Iterable[PipelineInterval],
        operation: str,
    ) -> Tuple[PipelineInterval, ...]:
        return tuple(item for item in intervals if item.operation == operation)

    @staticmethod
    def _append_overlap(
        output: List[PipelineOverlap],
        first: PipelineInterval,
        second: PipelineInterval,
    ) -> None:
        overlap_ns = min(first.end_ns, second.end_ns) - max(
            first.start_ns,
            second.start_ns,
        )
        if overlap_ns <= 0:
            return
        output.append(
            PipelineOverlap(
                phase=first.phase,
                first_operation=first.operation,
                second_operation=second.operation,
                first_identity=first.identity,
                second_identity=second.identity,
                first_start_ns=first.start_ns,
                first_end_ns=first.end_ns,
                second_start_ns=second.start_ns,
                second_end_ns=second.end_ns,
                overlap_ns=overlap_ns,
            )
        )
