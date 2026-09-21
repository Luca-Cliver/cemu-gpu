#!/usr/bin/env python3
"""Analyze CEMU host execution intervals; these are not target-system latency."""
import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path

from sparf_model_diagnostics import (join_completions, validate_completion,
    validate_diagnostics, summarize_model, print_model)


STAGES = {0: "legacy", 1: "select", 2: "token_read", 3: "exact_v", 4: "pv"}


def identity(record):
    return (record["program"], record["workflow"] if record["stage"] else record["command"])


def read_records(path, min_tokens=0):
    records = []
    completions = []
    with Path(path).open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("event") == "completion":
                    validate_completion(record)
                    completions.append(record)
                    continue
                validate_diagnostics(record)
                if record["schema"] != 1 or record["clock"] != "host_monotonic_ns":
                    raise ValueError("unsupported schema/clock")
                if record["stage"] not in STAGES:
                    raise ValueError("unknown stage")
                times = [record[k] for k in ("worker_start_ns", "start_ns", "end_ns", "worker_end_ns")]
                if any(not isinstance(t, int) or t < 0 for t in times) or times != sorted(times):
                    raise ValueError("invalid command timestamps")
                for event in record["intervals"]:
                    if not record["start_ns"] <= event["start_ns"] <= event["end_ns"] <= record["end_ns"]:
                        raise ValueError("phase outside command interval")
                for key in ("pid", "session", "controller", "program", "workflow", "command",
                            "layer", "tokens", "worker", "workers", "compute_units", "modeled_ns"):
                    if not isinstance(record[key], int) or record[key] < 0:
                        raise ValueError(f"invalid {key}")
                if not isinstance(record["success"], bool):
                    raise ValueError("invalid success flag")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            records.append(record)
    join_completions(records, completions)
    return [r for r in records if r["tokens"] >= min_tokens]


def interval(record, start=None, end=None):
    return (record["start_ns"] if start is None else start,
            record["end_ns"] if end is None else end,
            identity(record), record["layer"], record["tokens"])


def overlap_ns(left, right):
    """Union duration of cross-workflow overlap within the same layer/token.

    A sweep avoids counting the same wall interval repeatedly for nested spans
    or more than two active slots. Same-workflow V/QK nesting is excluded.
    """
    events = []
    for side, spans in enumerate((left, right)):
        for start, end, workflow, layer, tokens in spans:
            if end > start:
                key = (workflow, layer, tokens)
                events.extend(((start, side, key, 1), (end, side, key, -1)))
    active = (Counter(), Counter())
    previous = None
    total = 0
    for timestamp, changes in groupby(sorted(events), key=lambda event: event[0]):
        if previous is not None and any(a[0] != b[0] and a[1:] == b[1:]
                                        for a in active[0] for b in active[1]):
            total += timestamp - previous
        for _, side, key, delta in changes:
            active[side][key] += delta
            if active[side][key] == 0:
                del active[side][key]
        previous = timestamp
    return total


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        mode = "staged" if record["stage"] else "legacy"
        grouped[(record["pid"], record["session"], record["controller"], mode)].append(record)
    summaries = []
    for (pid, session, controller, mode), rows in sorted(grouped.items()):
        workflows = defaultdict(list)
        for row in rows:
            workflows[identity(row)].append(row)
        complete = 0
        for commands in workflows.values():
            ordered = sorted(commands, key=lambda r: r["stage"])
            stages = [r["stage"] for r in ordered]
            if len(stages) != len(set(stages)):
                raise ValueError("duplicate stage in one workflow")
            if len({(r["layer"], r["tokens"]) for r in ordered}) != 1:
                raise ValueError("workflow changed layer/token identity")
            if any(a["end_ns"] > b["start_ns"] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("dependent stages overlap or execute out of order")
            if stages == ([0] if mode == "legacy" else [1, 2, 3, 4]) and all(r["success"] for r in ordered):
                complete += 1
        loads = [interval(r) for r in rows if r["success"] and r["stage"] in (1, 2)]
        computes = [interval(r) for r in rows if r["success"] and r["stage"] in (3, 4)]
        reads, qks = [], []
        for r in rows:
            if not r["success"]:
                continue
            for event in r["intervals"]:
                if event["name"] in ("channel_read_wait", "token_k_read_wait"):
                    reads.append(interval(r, event["start_ns"], event["end_ns"]))
                elif event["name"] == "qk_functional":
                    qks.append(interval(r, event["start_ns"], event["end_ns"]))
        summaries.append(dict(
            pid=pid, session=session, controller=controller, mode=mode,
            model=summarize_model(rows),
            commands=len(rows), complete_workflows=complete,
            incomplete_or_failed_workflows=len(workflows) - complete,
            failed_commands=sum(not r["success"] for r in rows),
            workers=sorted({r["workers"] for r in rows}),
            compute_units=sorted({r["compute_units"] for r in rows}),
            load_compute_overlap_ns=overlap_ns(loads, computes),
            read_wait_qk_overlap_ns=overlap_ns(reads, qks),
            freeze_entry_sum_ns=sum(r["start_ns"] - r["worker_start_ns"] for r in rows),
            freeze_exit_sum_ns=sum(r["worker_end_ns"] - r["end_ns"] for r in rows),
        ))
    return summaries


def chrome_trace(records):
    origin = min((r["worker_start_ns"] for r in records), default=0)
    events = []
    processes = {}
    for r in records:
        group = (r["pid"], r["session"], r["controller"])
        if group not in processes:
            processes[group] = len(processes) + 1
        args = {k: r[k] for k in ("command", "program", "workflow", "layer", "tokens", "success")}
        spans = [dict(name=STAGES[r["stage"]], start_ns=r["start_ns"], end_ns=r["end_ns"])]
        for event in spans + r["intervals"]:
            events.append(dict(name=event["name"], cat="host_functional", ph="X",
                ts=(event["start_ns"] - origin) / 1000,
                dur=(event["end_ns"] - event["start_ns"]) / 1000,
                pid=processes[group], tid=r["worker"], args=args))
    return dict(traceEvents=events, displayTimeUnit="ns")


def virtual_trace(records):
    """A separate trace: never place host-monotonic and virtual timestamps together."""
    events = []
    processes = {}
    for r in records:
        key = (r["pid"], r["session"], r["controller"])
        processes.setdefault(key, len(processes) + 1)
        args = {k: r[k] for k in ("command", "program", "workflow", "stage", "layer", "tokens")}
        spans = []
        if "scheduler" in r:
            q = r["scheduler"]
            spans.append(("scheduler job (includes read waits)", f"CU {q['cu']}", q["start_ns"], q["finish_ns"]))
            spans.append(("received to scheduled (includes dispatch)", f"slot {r['program']}", q["received_ns"], q["start_ns"]))
        for read in r.get("reads", []):
            spans.append((read["name"] + " request (includes queue)",
                          f"NAND requests slot {r['program']}", read["model_start_ns"], read["model_end_ns"]))
        for name, lane, start, end in spans:
            events.append(dict(name=name, cat="qemu_virtual", ph="X", ts=start / 1000,
                dur=(end-start) / 1000, pid=processes[key], tid=lane, args=args))
    origin = min((e["ts"] for e in events), default=0)
    for e in events:
        e["ts"] -= origin
    return dict(traceEvents=events, displayTimeUnit="ns")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chrome-trace", type=Path)
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--virtual-trace", type=Path)
    parser.add_argument("--require-model", action="store_true",
                        help="require complete virtual scheduler/read diagnostics")
    args = parser.parse_args()
    try:
        records = read_records(args.timeline, args.min_tokens)
        if not records:
            raise ValueError("no matching timeline records")
        reports = summarize(records)
    except (OSError, ValueError) as error:
        parser.exit(1, f"[sparf-timeline] {error}\n")
    for report in reports:
        print(f"[sparf-timeline] controller={report['controller']} mode={report['mode']}: "
              f"commands={report['commands']}, complete={report['complete_workflows']}, "
              f"incomplete/failed={report['incomplete_or_failed_workflows']}, "
              f"workers={report['workers']}, compute_units={report['compute_units']}")
        print(f"  cross-workflow Load/Compute={report['load_compute_overlap_ns'] / 1e6:.6f}ms, "
              f"read-wait/QK={report['read_wait_qk_overlap_ns'] / 1e6:.6f}ms")
        print_model(report["model"])
    observed = any(r["load_compute_overlap_ns"] > 0 for r in reports)
    print("[sparf-timeline] cross-workflow host overlap: " + ("OBSERVED" if observed else "NOT OBSERVED"))
    print("[sparf-timeline] host execution only; read-wait includes host handling, "
          "V outstanding is not NAND service time, and these are not target-system latency/speedup.")
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if args.chrome_trace:
        args.chrome_trace.write_text(json.dumps(chrome_trace(records)) + "\n")
    if args.virtual_trace:
        args.virtual_trace.write_text(json.dumps(virtual_trace(records)) + "\n")
    print("[sparf-model] virtual scheduler intervals include NAND waits charged as job runtime; "
          "NAND intervals include queueing. Neither is isolated compute service. "
          "Per-page service/wait sums overlap; initial wait is LUN backlog at first access. "
          "Completion is device scheduler completion, not Guest ioctl return.")
    if args.require_model and any(not r["model"]["available"] or
            r["model"]["observed_commands"] != r["commands"] or
            r["model"]["scheduler"]["missing_completions"] or
            r["incomplete_or_failed_workflows"] for r in reports):
        parser.exit(2, "[sparf-model] diagnostics incomplete; finish the run or enable the new logger\n")
    if args.require_overlap and (not observed or any(r["incomplete_or_failed_workflows"] for r in reports)):
        parser.exit(2, "[sparf-timeline] overlap validation did not pass\n")


if __name__ == "__main__":
    main()
