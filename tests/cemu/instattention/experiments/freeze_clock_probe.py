"""Measure Guest clock leakage across CEMU functional freezing, without OPT."""

import argparse
import json
import os
from pathlib import Path
import secrets
import statistics
import struct
import sys
import tempfile
import time


MAGIC = 0x43454D55434C4B31
MARKER = "CEMU_CLOCK_TRACE "
DELAYS_MS = (0, 200, 1000)
CPP_CLOCK_FIELDS = ("cpp_steady_ns", "cpp_high_resolution_ns", "cpp_system_ns")


def measure_execute(execute, snapshot):
    before = time.perf_counter_ns()
    cpp_before = snapshot()
    status = execute()
    cpp_after = snapshot()
    elapsed = time.perf_counter_ns() - before
    return status, dict(guest_execute_ns=elapsed, **{
        field: end - start
        for field, start, end in zip(CPP_CLOCK_FIELDS, cpp_before, cpp_after)
    })


def run_probe(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from device_test_support import require_fdmfs
    from _cemu_client import CemuClient, MemoryRange, ProgramTarget, guest_clock_info, guest_clock_snapshot

    require_fdmfs(args.fdm_dir)
    if not 1 <= args.runtime_ms <= 2000 or not 1 <= args.repetitions <= 20:
        raise ValueError("runtime must be 1..2000 ms; repetitions must be 1..20")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_ns = args.runtime_ms * 1_000_000
    session = secrets.randbits(40) << 16
    report = dict(schema=2, backend=os.environ.get("CEMU_IO_BACKEND", "ioctl"),
                  clocksource=Path("/sys/devices/system/clocksource/clocksource0/current_clocksource").read_text().strip(),
                  perf_counter=time.get_clock_info("perf_counter").implementation,
                  cpp_clocks=guest_clock_info(), kernel=os.uname().release,
                  kernel_command_line=Path("/proc/cmdline").read_text().strip(),
                  runtime_ns=runtime_ns, rows=[])
    baseline = [measure_execute(lambda: 0, guest_clock_snapshot)[1] for _ in range(20)]
    report["empty_bracket_median_ns"] = {
        field: statistics.median(row[field] for row in baseline)
        for field in ("guest_execute_ns", *CPP_CLOCK_FIELDS)
    }
    print(f"[freeze-clock] {report['clocksource']=}, {report['backend']=}, runtime={args.runtime_ms}ms", flush=True)
    print(f"[freeze-clock] Python={report['perf_counter']}, C++={report['cpp_clocks']}", flush=True)
    print(f"[freeze-clock] empty bracket medians (ns)={report['empty_bracket_median_ns']}; not subtracted", flush=True)
    print("[freeze-clock] Run alone: all Guest vCPUs pause during each Host operator sleep.", flush=True)
    with output.open("x", encoding="utf-8") as destination:
        with tempfile.TemporaryDirectory(prefix="freeze-clock-", dir=args.fdm_dir) as directory:
            with CemuClient(args.control, args.namespace) as client:
                client.load_program(f"clock_probe_{session:x}", args.program, "freeze_clock_probe",
                                    ProgramTarget.HOST, runtime_ns, 0, False, False)
                client.activate_program()
                client.create_memory_ranges([MemoryRange(str(Path(directory) / "result"), 0, 4096)])
                report["program"] = client.program_id
                if client.execute(cparam1=0, cparam2=session, runtime=runtime_ns):
                    raise RuntimeError("probe warmup failed")
                for repetition in range(args.repetitions):
                    delays = DELAYS_MS if repetition % 2 == 0 else tuple(reversed(DELAYS_MS))
                    for delay_ms in delays:
                        probe_id = session + len(report["rows"]) + 1
                        status, measurements = measure_execute(
                            lambda: client.execute(cparam1=delay_ms * 1_000_000,
                                                   cparam2=probe_id, runtime=runtime_ns),
                            guest_clock_snapshot)
                        if status:
                            raise RuntimeError(f"probe execute returned {status}")
                        magic, returned_id, delay_ns, host_start, host_end = struct.unpack_from(
                            "<5Q", client.read_range_bytes(0, 4096))
                        if (magic != MAGIC or returned_id != probe_id or delay_ns != delay_ms * 1_000_000 or
                                host_end < host_start or host_end - host_start < delay_ns):
                            raise RuntimeError("invalid or stale probe output")
                        row = dict(probe_id=probe_id, delay_ms=delay_ms, repetition=repetition,
                                   **measurements, host_operator_ns=host_end - host_start)
                        report["rows"].append(row)
                        print(f"[freeze-clock] delay={delay_ms}ms, Guest Execute={row['guest_execute_ns'] / 1e6:.3f}ms, "
                              f"C++ steady={row['cpp_steady_ns'] / 1e6:.3f}ms, "
                              f"C++ high_resolution={row['cpp_high_resolution_ns'] / 1e6:.3f}ms, "
                              f"C++ system={row['cpp_system_ns'] / 1e6:.3f}ms, "
                              f"Host operator={(host_end - host_start) / 1e6:.3f}ms", flush=True)
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"[freeze-clock] Measurements saved: {output}; not a timing-model accuracy PASS.")
    print("[freeze-clock] Analyze with the same QEMU session log to verify virtual time.")


def analyze(report, lines):
    rows = report["rows"]
    wanted = {row["probe_id"] for row in rows}
    if not rows or len(wanted) != len(rows):
        raise ValueError("empty or duplicate Guest probe IDs")
    events = {}
    decoder = json.JSONDecoder()
    for line in lines:
        if MARKER not in line:
            continue
        event, _ = decoder.raw_decode(line.split(MARKER, 1)[1].lstrip())
        if event["probe_id"] not in wanted:
            continue
        key = event["probe_id"], event["event"]
        if key in events:
            raise ValueError(f"duplicate Host event: {key}")
        events[key] = event
    merged = []
    identities = set()
    for row in rows:
        probe_id = row["probe_id"]
        if any((probe_id, event) not in events for event in ("functional", "completion")):
            raise ValueError(f"missing Host samples for {probe_id}; enable CEMU_COMPUTE_CLOCK_TRACE=1 and use the matching session")
        functional = events[probe_id, "functional"]
        completion = events[probe_id, "completion"]
        identity = tuple(functional[field] for field in ("pid", "controller", "program"))
        if identity != tuple(completion[field] for field in ("pid", "controller", "program")) or identity[2] != report["program"]:
            raise ValueError("Host/Guest program or session mismatch")
        identities.add(identity)
        if functional["requested_runtime_ns"] != report["runtime_ns"]:
            raise ValueError("requested runtime changed during experiment")
        frozen_ns = functional["virtual_end_ns"] - functional["virtual_start_ns"]
        functional_ns = functional["host_end_ns"] - functional["host_start_ns"]
        device_ns = completion["finish_virtual_ns"] - completion["received_virtual_ns"]
        if min(frozen_ns, functional_ns, device_ns) < 0:
            raise ValueError("non-monotonic timestamps")
        merged.append(dict(row, frozen_virtual_ns=frozen_ns, host_functional_ns=functional_ns,
                           device_virtual_ns=device_ns, effective_runtime_ns=functional["effective_runtime_ns"],
                           scheduled_runtime_ns=completion["scheduled_runtime_ns"],
                           freeze_entry_ns=functional["freeze_entry_ns"]))
    if len(identities) != 1:
        raise ValueError("mixed QEMU sessions")
    summaries = []
    cpp_fields = tuple(field for field in CPP_CLOCK_FIELDS if all(field in row for row in merged))
    for delay_ms in DELAYS_MS:
        selected = [row for row in merged if row["delay_ms"] == delay_ms]
        if not selected:
            raise ValueError(f"missing delay={delay_ms}ms case")
        summaries.append(dict(delay_ms=delay_ms, samples=len(selected), **{
            field: statistics.median(row[field] for row in selected)
            for field in ("guest_execute_ns", "host_functional_ns", "device_virtual_ns",
                          "effective_runtime_ns", "scheduled_runtime_ns", "freeze_entry_ns", *cpp_fields)}))
    base, slow = summaries[0], summaries[-1]
    host_growth = slow["host_functional_ns"] - base["host_functional_ns"]
    guest_growth = slow["guest_execute_ns"] - base["guest_execute_ns"]
    device_growth = slow["device_virtual_ns"] - base["device_virtual_ns"]
    frozen_max = max(row["frozen_virtual_ns"] for row in merged)
    budgets_valid = all(0 < row["effective_runtime_ns"] <= report["runtime_ns"] and
                        row["scheduled_runtime_ns"] == row["effective_runtime_ns"] for row in merged)
    ratio = guest_growth / host_growth if host_growth > 0 else None
    verdict = "inconclusive"
    if frozen_max > 1000:
        verdict = "virtual_clock_advanced_inside_freeze"
    elif budgets_valid and host_growth >= 100_000_000 and abs(device_growth) <= 0.2 * host_growth:
        if 0.8 <= ratio <= 1.2:
            verdict = "guest_clock_includes_functional_freeze"
        elif abs(ratio) <= 0.2:
            verdict = "guest_clock_excludes_functional_freeze"
    return dict(verdict=verdict, guest_growth_per_host_growth=ratio,
                cpp_growth_per_host_growth={
                    field: (slow[field] - base[field]) / host_growth if host_growth > 0 else None
                    for field in cpp_fields},
                environment={key: report.get(key) for key in (
                    "clocksource", "perf_counter", "cpp_clocks", "kernel", "kernel_command_line",
                    "backend", "empty_bracket_median_ns")},
                frozen_virtual_max_ns=frozen_max, budgets_valid=budgets_valid,
                summaries=summaries, rows=merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--program", required=True, help="shared library path as seen by Host QEMU")
    run.add_argument("--output", required=True, help="new Guest JSON file (never overwritten)")
    run.add_argument("--fdm-dir", default="/mnt/fdm0")
    run.add_argument("--control", default="/dev/nvme0c3")
    run.add_argument("--namespace", default="/dev/ng0n3")
    run.add_argument("--runtime-ms", type=int, default=100)
    run.add_argument("--repetitions", type=int, default=3)
    review = commands.add_parser("analyze")
    review.add_argument("--guest-json", required=True)
    review.add_argument("--host-log", required=True)
    review.add_argument("--output")
    args = parser.parse_args()
    if args.command == "run":
        run_probe(args)
    else:
        report = json.loads(Path(args.guest_json).read_text())
        with open(args.host_log, encoding="utf-8", errors="replace") as source:
            result = analyze(report, source)
        for row in result["summaries"]:
            print(f"delay={row['delay_ms']}ms: Guest={row['guest_execute_ns'] / 1e6:.3f}ms, "
                  f"Host functional={row['host_functional_ns'] / 1e6:.3f}ms, "
                  f"device virtual={row['device_virtual_ns'] / 1e6:.3f}ms, "
                  f"scheduled={row['scheduled_runtime_ns'] / 1e6:.3f}ms")
            if all(field in row for field in CPP_CLOCK_FIELDS):
                print("  " + ", ".join(f"{field}={row[field] / 1e6:.3f}ms" for field in CPP_CLOCK_FIELDS))
        print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "summaries")}, indent=2))
        print("Device interval is request receipt to scheduler completion, not Guest return. No cross-clock subtraction.")
        if args.output:
            with open(args.output, "x", encoding="utf-8") as destination:
                json.dump(result, destination, indent=2)
                destination.write("\n")


if __name__ == "__main__":
    main()
