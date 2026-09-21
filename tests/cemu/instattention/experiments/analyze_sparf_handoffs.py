"""Separate stage handoffs from device admission without mixing clock domains."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from analyze_sparf_timeline import identity, read_records, STAGES


def covered_ns(start, end, spans):
    cursor = start
    covered = 0
    for left, right in sorted(spans):
        left, right = max(start, left, cursor), min(end, right)
        if right > left:
            covered += right - left
            cursor = right
    return covered


def admission_summary(rows):
    totals = defaultdict(int)
    for row in rows:
        if "dispatch" not in row:
            continue
        dispatch = row["dispatch"]
        if (dispatch["host_clock"] != "host_monotonic_ns" or
                dispatch["virtual_clock"] != "qemu_virtual_ns"):
            raise ValueError("unsupported dispatch clocks")
        host = [dispatch[name] for name in ("host_entry_ns", "host_mirror_start_ns",
                "host_mirror_end_ns", "host_enqueued_ns")]
        scheduler = row["scheduler"]
        virtual = [scheduler["received_ns"], dispatch["virtual_entry_ns"],
                   dispatch["virtual_enqueued_ns"], scheduler["start_ns"]]
        for times in (host, virtual):
            if any(type(value) is not int or value < 0 for value in times) or times != sorted(times):
                raise ValueError("invalid dispatch timestamps")
        totals["observed_commands"] += 1
        totals["host_mirror_prepare_ns"] += host[2] - host[1]
        totals["host_other_dispatch_ns"] += host[1] - host[0] + host[3] - host[2]
        totals["virtual_received_to_handler_ns"] += virtual[1] - virtual[0]
        totals["virtual_handler_to_enqueued_ns"] += virtual[2] - virtual[1]
        totals["virtual_enqueued_to_start_ns"] += virtual[3] - virtual[2]
    return dict(totals) if totals else None


def summarize_handoffs(records, top=5):
    sessions = defaultdict(list)
    for row in records:
        if row["success"] and "scheduler" in row:
            sessions[row["pid"], row["session"], row["controller"]].append(row)
    reports = []
    for session, rows in sorted(sessions.items()):
        workflows = defaultdict(list)
        for row in rows:
            workflows[identity(row)].append(row)
        gaps = []
        by_stage = defaultdict(list)
        for workflow in workflows.values():
            ordered = sorted(workflow, key=lambda row: row["stage"])
            for left, right in zip(ordered, ordered[1:]):
                if right["stage"] != left["stage"] + 1:
                    continue
                start = left["scheduler"]["finish_ns"]
                end = right["scheduler"]["received_ns"]
                if end < start:
                    raise ValueError("next stage arrived before previous completion")
                name = f"{STAGES[left['stage']]}->{STAGES[right['stage']]}"
                by_stage[name].append(end - start)
                gaps.append(dict(program=left["program"], workflow=left["workflow"],
                    layer=left["layer"], tokens=left["tokens"], transition=name,
                    previous_command=left["command"], next_command=right["command"],
                    start_ns=start, end_ns=end, duration_ns=end-start))
        longest = sorted(gaps, key=lambda gap: gap["duration_ns"], reverse=True)[:top]
        for gap in longest:
            start, end = gap["start_ns"], gap["end_ns"]
            peers = [row for row in rows if identity(row) != (gap["program"], gap["workflow"])
                     and (row["layer"], row["tokens"]) == (gap["layer"], gap["tokens"])]
            gap["peer_received_to_start_cover_ns"] = covered_ns(start, end, (
                (row["scheduler"]["received_ns"], row["scheduler"]["start_ns"])
                for row in peers))
            gap["observed_cu_job_cover_ns"] = covered_ns(start, end, (
                (row["scheduler"]["start_ns"], row["scheduler"]["finish_ns"])
                for row in rows))
            gap["peers"] = [{**row["scheduler"], "program": row["program"],
                "stage": STAGES[row["stage"]]}
                for row in peers if row["scheduler"]["received_ns"] < end
                and row["scheduler"]["finish_ns"] > start]
        reports.append(dict(pid=session[0], session=session[1], controller=session[2],
            clock="qemu_virtual_ns", programs=sorted({row["program"] for row in rows}),
            completed_commands=len(rows), missing_completions=sum(
                "scheduler" not in row for row in records
                if (row["pid"], row["session"], row["controller"]) == session),
            stages={name: dict(count=len(values), total_ns=sum(values),
                median_ns=median(values), max_ns=max(values)) for name, values in by_stage.items()},
            longest=longest, admission=admission_summary(rows)))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--programs", type=int, nargs="+")
    parser.add_argument("--min-tokens", type=int, default=1024)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        rows = read_records(args.timeline, args.min_tokens)
        if args.programs is not None:
            rows = [row for row in rows if row["program"] in args.programs]
        reports = summarize_handoffs(rows, args.top)
        if not reports:
            raise ValueError("no matching completed commands")
        if args.output:
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"[sparf-handoffs] {error}\n")
    for report in reports:
        print(f"[sparf-handoffs] session={report['session']} programs={report['programs']} "
              f"completed={report['completed_commands']} missing={report['missing_completions']}")
        for name, values in report["stages"].items():
            print(f"  {name}: count={values['count']}, sum={values['total_ns']/1e9:.6f}s, "
                  f"median={values['median_ns']/1e6:.6f}ms, max={values['max_ns']/1e6:.6f}ms")
        for gap in report["longest"]:
            print(f"  gap program={gap['program']} workflow={gap['workflow']} "
                  f"layer={gap['layer']} tokens={gap['tokens']} {gap['transition']}: "
                  f"gap={gap['duration_ns']/1e6:.6f}ms, "
                  f"peer pre-start cover={gap['peer_received_to_start_cover_ns']/1e6:.6f}ms, "
                  f"observed CU job cover={gap['observed_cu_job_cover_ns']/1e6:.6f}ms")
        print("  admission:", json.dumps(report["admission"], sort_keys=True))
    print("[sparf-handoffs] pre-start includes dispatch/mirror preparation, NOT just CU queueing. "
          "Coverage is correlation, not proof of causation; CU job spans include read waits. "
          "Gap sums may overlap. Missing admission records cannot locate time inside pre-start.")


if __name__ == "__main__":
    main()
