"""Timing reports keep Guest wall, CUDA stream, and Host device clocks separate."""

from dataclasses import asdict


def batch_phase(prefill):
    flags = set(bool(value) for value in prefill)
    return {frozenset({True}): "prefill", frozenset({False}): "decode",
            frozenset({True, False}): "mixed"}.get(frozenset(flags), "empty")


def timing_report(profiler):
    if profiler is None:
        return {"enabled": False}
    metrics = {item.name: asdict(item) for item in profiler.snapshot()}
    cuda_diagnostics = {name: metrics.pop(name) for name in list(metrics)
                        if name.endswith(".cuda_stream")}

    def total(name):
        return metrics.get(name, {}).get("total_ns", 0)

    phases = {}
    for phase in ("prefill", "decode", "mixed"):
        name = phase + ".step_wall"
        item = metrics.get(name, {})
        phases[phase] = {
            "steps": item.get("count", 0), "seconds": total(name) / 1e9,
            "average_step_ms": total(name) / max(item.get("count", 0), 1) / 1e6,
        }
    # These are nested metrics, not an additive partition of generate().
    step_ns = sum(total(p + ".step_wall") for p in phases)
    exec_ns = total("decode.execute_wait_wall")
    return {
        "enabled": True, "detailed": profiler.stack, "phases": phases,
        "exec_wait_seconds": exec_ns / 1e9,
        "exec_wait_percent_of_worker_steps": 100 * exec_ns / step_ns if step_ns else None,
        "metrics": metrics, "cuda_diagnostics": cuda_diagnostics,
        "metadata": profiler.metadata_snapshot(),
        "clock": "Guest CLOCK_MONOTONIC with HPET required by run.py",
        "semantics": {
            "step_wall": "execute_model entry through synchronous sample_tokens return; includes RPC gap, excludes frontend scheduling outside this interval",
            "execute_host_wall": "CPU call duration; GPU work can finish during sample_tokens",
            "exec_wait": "blocking CEMU command wait including Guest/transport/device; not pure GPU compute",
            "freeze": "HPET Guest time excludes functional freeze; never subtract Host functional time again",
            "cuda_stream": "separate GPU clock can include CEMU freezes; diagnostics only, excluded from main metrics and percentages",
            "software": "binding/buffer/packing/mean scopes measure caller wall, not pure CPU time; unmeasured residual is not software overhead",
            "cold_start": "first real prefill includes lazy CEMU registration and pinned buffer allocation",
        },
    }


def union_ns(spans):
    total, end = 0, None
    for left, right in sorted(spans):
        if left < 0 or right < left:
            raise ValueError("invalid device interval")
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def device_report(records, program, expected):
    """Use this run's complete commands on the freeze-aware QEMU virtual clock.

    The caller reads only bytes appended since generate() began. Never divide
    QEMU virtual durations by Guest wall time, or sum overlapping CU spans.
    """
    rows = [r for r in records if r.get("schema") == "sparf-autonomous-v1"
            and r.get("program") == program]
    begins, completes = {}, {}
    key = lambda r: tuple(r[k] for k in ("pid", "session", "controller", "request", "program"))
    for row in rows:
        destination = begins if row.get("event") == "begin" else completes
        if row.get("event") not in ("begin", "complete"):
            raise ValueError("unknown device timeline event")
        identity = key(row)
        if identity in destination:
            raise ValueError("duplicate device timeline event")
        destination[identity] = row
    if len(completes) != expected or begins.keys() != completes.keys():
        return {"available": False, "reason": "device command count/completion mismatch",
                "expected": expected, "begins": len(begins), "completes": len(completes)}
    if not completes:
        return {"available": False, "reason": "no decode EXEC commands"}
    if len({key(r)[:3] for r in completes.values()}) != 1:
        raise ValueError("device timeline contains multiple controller sessions")
    spans, functions, reads, virtual = [], {}, {}, 0
    host_spans, host_reads, compute_spans = [], {}, {}
    zero_budget = 0
    for row in completes.values():
        if not row["success"]:
            raise ValueError("failed device command")
        begin = begins[key(row)]
        if begin["admitted"] != row["admitted"]:
            raise ValueError("device admission mismatch")
        if row.get("host_clock") != "host_monotonic_ns" or row.get("virtual_clock") != "qemu_virtual_ns":
            raise ValueError("unsupported device clocks")
        host_left, host_right = row["admitted"]["host_ns"], row["ready"]["host_ns"]
        host_spans.append((host_left, host_right))
        left, right = row["admitted"]["virtual_ns"], row["ready"]["virtual_ns"]
        spans.append((left, right))
        delta = right - left
        if delta < 0:
            raise ValueError("invalid virtual interval")
        virtual += delta
        for phase in row["reads"]:
            a, b = phase["submitted"]["virtual_ns"], phase["ready"]["virtual_ns"]
            if not left <= a <= b <= right:
                raise ValueError("read interval outside device command")
            reads.setdefault(phase["name"], []).append((a, b))
            a, b = phase["submitted"]["host_ns"], phase["ready"]["host_ns"]
            if not host_left <= a <= b <= host_right:
                raise ValueError("Host read interval outside device command")
            host_reads.setdefault(phase["name"], []).append((a, b))
        for phase in row["computes"]:
            a, b = phase["functional_start"]["host_ns"], phase["functional_end"]["host_ns"]
            if not host_left <= a <= b <= host_right:
                raise ValueError("functional interval outside device command")
            if phase["functional_end"]["virtual_ns"] != phase["functional_start"]["virtual_ns"]:
                raise ValueError("virtual clock advanced during functional freeze")
            functions.setdefault(phase["name"], []).append((a, b))
            zero_budget += phase["effective_ns"] == 0
            if phase.get("truncated"):
                raise ValueError("truncated modeled compute spans")
            compute_spans.setdefault(phase["name"], [])
            for span in phase["cu_spans"]:
                a, b = span["start_ns"], span["end_ns"]
                if not left <= a <= b <= right:
                    raise ValueError("modeled compute interval outside device command")
                compute_spans.setdefault(phase["name"], []).append((a, b))
    busy = union_ns(spans)
    window = max(b for _, b in spans) - min(a for a, _ in spans)
    functional = union_ns([s for values in functions.values() for s in values])
    compute = union_ns([s for values in compute_spans.values() for s in values])
    read = union_ns([s for values in reads.values() for s in values])
    covered = union_ns([s for values in (*compute_spans.values(), *reads.values()) for s in values])
    host_busy = union_ns(host_spans)
    host_window = max(b for _, b in host_spans) - min(a for a, _ in host_spans)
    return {
        "available": True, "commands": expected, "clock": "qemu_virtual_ns",
        "functional_freeze_excluded": True,
        "decode_command_window_seconds": window / 1e9,
        "device_command_union_seconds": busy / 1e9,
        "device_percent_of_decode_command_window": 100 * busy / window if window else None,
        "compute_union_seconds": compute / 1e9,
        "compute_percent_of_device_commands": 100 * compute / busy if busy else None,
        "compute_seconds_by_stage": {k: union_ns(v) / 1e9 for k, v in compute_spans.items()},
        "zero_budget_compute_phases": zero_budget,
        "read_submit_to_ready_seconds_by_stage": {k: union_ns(v) / 1e9 for k, v in reads.items()},
        "read_compute_overlap_seconds": (read + compute - covered) / 1e9,
        "unattributed_virtual_seconds": (busy - covered) / 1e9,
        "virtual_command_sum_seconds": virtual / 1e9,
        "host_diagnostics": {
            "clock": "host_monotonic_ns", "includes_functional_freeze": True,
            "command_union_seconds": host_busy / 1e9,
            "command_window_seconds": host_window / 1e9,
            "functional_union_seconds": functional / 1e9,
            "functional_seconds_by_stage": {k: union_ns(v) / 1e9 for k, v in functions.items()},
            "read_submit_to_ready_seconds_by_stage": {k: union_ns(v) / 1e9 for k, v in host_reads.items()},
            "semantics": "debug only; functional spans are NOT complete global freeze intervals; never subtract them from Guest time",
        },
        "semantics": "All primary durations/ratios use QEMU virtual time, excluding functional freeze; window is first EXEC admission to last ready, not whole generate; CU spans are modeled compute, reads include wait/polling and overlap compute; unattributed includes queues/dispatch, not pure CPU time",
    }


def print_timing(report):
    if not report.get("enabled"):
        return
    print("[timing] Worker steps (Guest HPET, functional freeze excluded; execute + sampling):")
    for name, phase in report["phases"].items():
        print(f"  {name}: {phase['seconds']:.6f}s, {phase['steps']} steps, "
              f"avg {phase['average_step_ms']:.3f} ms/step")
    ratio = report["exec_wait_percent_of_worker_steps"]
    if ratio is not None:
        print(f"[timing] CEMU EXEC wait: {report['exec_wait_seconds']:.6f}s "
              f"({ratio:.2f}% of worker steps; includes transport/device wait)")
    for name, metric in report["metrics"].items():
        print(f"  {name}: {metric['total_ns'] / 1e9:.6f}s ({metric['count']} calls)")
    if report["cuda_diagnostics"]:
        print("[diagnostic] CUDA stream times (GPU clock can include freeze; not part of main timings):")
        for name, metric in report["cuda_diagnostics"].items():
            print(f"  {name}: {metric['total_ns'] / 1e9:.6f}s")
    print("[timing] Metrics are nested/overlapping; do not sum them as a partition.")
