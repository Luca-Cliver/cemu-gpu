"""Validate v3 dependencies using separate Host and CEMU virtual clocks."""
import argparse
import json
from pathlib import Path


SCHEMA = "sparf-autonomous-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    require(type(value) is int and value >= 0, "invalid timestamp")
    return value


def stamp(record, name):
    value = record[name]
    return timestamp(value["host_ns"]), timestamp(value["virtual_ns"])


def ordered(values, label):
    require(values == sorted(values), f"{label}: timestamps out of order")


def overlap(left, right):
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def read_diagnostics(read):
    if "io" not in read:
        return None
    data = read["io"]
    fields = ("requested_bytes", "logical_bytes", "tail_bytes", "tail_copies", "logical_spans",
              "aligned_bytes_before_dedup", "submitted_bytes", "ranges", "model_start_ns",
              "model_end_ns", "pages", "unmapped_pages", "service_ns", "queue_ns", "max_queue_ns",
              "critical_queue_ns", "critical_service_ns", "critical_initial_wait_ns",
              "channels", "luns_per_channel", "page_bytes", "page_read_ns")
    for name in fields:
        timestamp(data[name])
    counts, initial = data["lun_pages"], data["lun_initial_wait_ns"]
    require(len(counts) == len(initial) == data["channels"] * data["luns_per_channel"],
            "inconsistent LUN geometry")
    for value in counts + initial:
        timestamp(value)
    require(sum(counts) == data["pages"], "inconsistent LUN page counts")
    require(all(count or not wait for count, wait in zip(counts, initial)), "wait on unused LUN")
    require(data["requested_bytes"] == data["logical_bytes"] + data["tail_bytes"],
            "requested bytes do not match Flash + tail")
    require(data["aligned_bytes_before_dedup"] >= data["submitted_bytes"], "invalid dedup bytes")
    require(data["critical_initial_wait_ns"] <= data["critical_queue_ns"], "invalid initial wait")
    require(data["model_start_ns"] == stamp(read, "submitted")[1] and
            data["model_end_ns"] == read["deadline_ns"], "IO interval differs from read deadline")
    if data["pages"]:
        require(data["page_bytes"] > 0 and data["page_read_ns"] > 0, "missing NAND geometry")
    result = dict(data)
    result.update(
        model_ns=data["model_end_ns"] - data["model_start_ns"],
        host_submit_to_observed_ns=stamp(read, "observed")[0] - stamp(read, "submitted")[0],
        host_observed_to_ready_ns=stamp(read, "ready")[0] - stamp(read, "observed")[0],
        virtual_deadline_to_ready_ns=stamp(read, "ready")[1] - read["deadline_ns"],
        critical_subsequent_queue_ns=data["critical_queue_ns"] - data["critical_initial_wait_ns"],
        dedup_saved_bytes=data["aligned_bytes_before_dedup"] - data["submitted_bytes"],
        active_luns=sum(count > 0 for count in counts),
        channel_pages=[sum(counts[offset:offset + data["luns_per_channel"]])
                       for offset in range(0, len(counts), data["luns_per_channel"] or 1)],
    )
    return result


def summarize_reads(reports):
    summary = {}
    for phase in ("channel_k", "token_k", "token_v"):
        items = [row["read_io"][phase] for row in reports if row["read_io"][phase] is not None]
        totals = {name: sum(item[name] for item in items) for name in (
            "requested_bytes", "logical_bytes", "tail_bytes", "submitted_bytes", "pages",
            "unmapped_pages", "dedup_saved_bytes", "model_ns", "service_ns", "queue_ns",
            "critical_initial_wait_ns", "critical_subsequent_queue_ns", "critical_service_ns",
            "host_submit_to_observed_ns", "host_observed_to_ready_ns", "virtual_deadline_to_ready_ns")}
        totals.update(observed_requests=len(items), missing_requests=len(reports) - len(items))
        totals["tail_fraction"] = totals["tail_bytes"] / totals["requested_bytes"] if totals["requested_bytes"] else None
        totals["submitted_to_logical_ratio"] = totals["submitted_bytes"] / totals["logical_bytes"] if totals["logical_bytes"] else None
        totals["active_luns_mean"] = sum(item["active_luns"] for item in items) / len(items) if items else None
        totals["lun_max_to_mean"] = max(
            max(item["lun_pages"]) * len(item["lun_pages"]) / item["pages"]
            for item in items if item["pages"]
        ) if any(item["pages"] for item in items) else None
        summary[phase] = totals
    return summary


def select_records(records, latest_session=False, min_tokens=0, max_tokens=None):
    if latest_session:
        begins = [row for row in records if row.get("event") == "begin"]
        require(bool(begins), "no session begin records")
        latest = begins[-1]
        records = [row for row in records if (row["pid"], row["session"]) == (latest["pid"], latest["session"])]
    return [row for row in records if row["tokens"] >= min_tokens and
            (max_tokens is None or row["tokens"] <= max_tokens)]


def validate_request(row):
    require(row["success"] is True, "device request failed")
    require(row["host_clock"] == "host_monotonic_ns" and
            row["virtual_clock"] == "qemu_virtual_ns", "unsupported clocks")
    admitted, started, ready = (stamp(row, name) for name in ("admitted", "started", "ready"))
    for clock in (0, 1):
        ordered([admitted[clock], started[clock], ready[clock]], "parent")
    require(len(row["reads"]) == 3 and len(row["computes"]) == 3, "missing stages")
    reads = {item["name"]: item for item in row["reads"]}
    computes = {item["name"]: item for item in row["computes"]}
    require(set(reads) == {"channel_k", "token_k", "token_v"}, "invalid read stages")
    require(set(computes) == {"approx", "qk", "pv"}, "invalid compute stages")
    for name, read in reads.items():
        submitted, observed, available = (stamp(read, field) for field in ("submitted", "observed", "ready"))
        for clock in (0, 1):
            ordered([started[clock], submitted[clock], observed[clock], available[clock], ready[clock]], name)
        ordered([submitted[1], timestamp(read["deadline_ns"]), available[1]], name + " deadline")
    spans = []
    for name, compute in computes.items():
        require(compute["truncated"] is False, f"{name}: CU trace truncated")
        require(type(compute["scheduled"]) is bool, "invalid scheduled flag")
        start, end, observed = (stamp(compute, field) for field in
                                ("functional_start", "functional_end", "observed_done"))
        for clock in (0, 1):
            ordered([started[clock], start[clock], end[clock], observed[clock], ready[clock]], name)
        submitted = timestamp(compute["submitted_ns"])
        completed = timestamp(compute["completed_ns"])
        effective = timestamp(compute["effective_ns"])
        ordered([end[1], submitted, completed, observed[1]], name + " scheduler")
        entries = compute["cu_spans"]
        if compute["scheduled"]:
            require(effective > 0 and bool(entries), f"{name}: missing CU occupancy")
        else:
            require(effective == 0 and not entries and submitted == completed,
                    f"{name}: invalid zero-budget bypass")
        duration = 0
        for entry in entries:
            require(type(entry["cu"]) is int and entry["cu"] >= 0, "invalid CU id")
            left, right = timestamp(entry["start_ns"]), timestamp(entry["end_ns"])
            ordered([submitted, left, right, completed], name + " CU span")
            duration += right - left
            spans.append((left, right, entry["cu"], name))
        require(duration >= effective, f"{name}: incomplete CU occupancy budget")

    def read_ready(name):
        return stamp(reads[name], "ready")[1]

    def compute_done(name):
        return stamp(computes[name], "observed_done")[1]

    require(stamp(computes["approx"], "functional_start")[1] >= read_ready("channel_k"),
            "approx started before channel K ready")
    require(stamp(reads["token_k"], "submitted")[1] >= compute_done("approx"),
            "token K read started before approximate compute completed")
    require(stamp(computes["qk"], "functional_start")[1] >= read_ready("token_k"),
            "QK started before selected K ready")
    require(stamp(reads["token_v"], "submitted")[1] >= read_ready("token_k"),
            "V read started before selected K ready")
    require(stamp(computes["pv"], "functional_start")[1] >=
            max(compute_done("qk"), read_ready("token_v")), "PV started before QK/V join")
    pure_waits = [(stamp(reads[name], "submitted")[1], read_ready(name))
                  for name in ("channel_k", "token_k")]
    pure_waits.append((max(compute_done("qk"), stamp(reads["token_v"], "submitted")[1]),
                       read_ready("token_v")))
    for left, right, _, name in spans:
        require(all(overlap((left, right), wait) == 0 for wait in pure_waits),
                f"{name}: parent occupies CU during pure IO wait")
    ordered_spans = sorted(spans)
    for previous, current in zip(ordered_spans, ordered_spans[1:]):
        require(previous[1] <= current[0], "one request has overlapping compute stages")
    value_read = (stamp(reads["token_v"], "submitted")[1], reads["token_v"]["deadline_ns"])
    qk_v_overlap = sum(overlap((left, right), value_read)
                       for left, right, _, name in spans if name == "qk")
    return {
        "request": row["request"], "layer": row["layer"], "tokens": row["tokens"],
        "host_elapsed_ns": ready[0] - admitted[0],
        "virtual_elapsed_ns": ready[1] - admitted[1],
        "engine_queue_host_ns": started[0] - admitted[0],
        "engine_queue_virtual_ns": started[1] - admitted[1],
        "read_boundaries": {
            name: {
                "model_ns": read["deadline_ns"] - stamp(read, "submitted")[1],
                "host_submit_to_observed_ns": stamp(read, "observed")[0] - stamp(read, "submitted")[0],
                "host_observed_to_ready_ns": stamp(read, "ready")[0] - stamp(read, "observed")[0],
                "virtual_deadline_to_ready_ns": stamp(read, "ready")[1] - read["deadline_ns"],
            } for name, read in reads.items()
        },
        "compute_boundaries": {
            name: {
                "functional_host_ns": stamp(compute, "functional_end")[0] - stamp(compute, "functional_start")[0],
                "functional_virtual_ns": stamp(compute, "functional_end")[1] - stamp(compute, "functional_start")[1],
                "end_to_submit_virtual_ns": compute["submitted_ns"] - stamp(compute, "functional_end")[1],
                "submit_to_complete_virtual_ns": compute["completed_ns"] - compute["submitted_ns"],
                "complete_to_observed_virtual_ns": stamp(compute, "observed_done")[1] - compute["completed_ns"],
            } for name, compute in computes.items()
        },
        "cu_occupied_ns": sum(right - left for left, right, _, _ in spans),
        "qk_v_target_overlap_ns": qk_v_overlap,
        "zero_budget_phases": sum(not item["scheduled"] for item in computes.values()),
        "read_io": {name: read_diagnostics(read) for name, read in reads.items()},
    }, spans


def summarize_boundaries(reports):
    summary = {
        "requests": len(reports),
        **{name: sum(row[name] for row in reports) for name in (
            "host_elapsed_ns", "virtual_elapsed_ns", "engine_queue_host_ns", "engine_queue_virtual_ns")},
    }
    for category in ("read_boundaries", "compute_boundaries"):
        summary[category] = {}
        for row in reports:
            for phase, fields in row[category].items():
                totals = summary[category].setdefault(phase, {})
                for name, value in fields.items():
                    totals[name] = totals.get(name, 0) + value
    return summary


def analyze(records):
    pending, seen, reports, occupancy, engines = {}, set(), [], {}, {}
    for row in records:
        require(row.get("schema") == SCHEMA, "wrong schema; use the dedicated v3 trace")
        identity = tuple(row[name] for name in ("pid", "session", "controller", "request"))
        if row["event"] == "begin":
            require(identity not in seen, "duplicate request begin")
            seen.add(identity)
            pending[identity] = row
        elif row["event"] == "complete":
            require(identity in pending, "missing begin or duplicate completion")
            begin = pending.pop(identity)
            for field in ("admitted", "program", "layer", "tokens", "host_clock", "virtual_clock"):
                require(begin[field] == row[field], f"changed request field: {field}")
            report, spans = validate_request(row)
            report["identity"] = identity
            reports.append(report)
            if row.get("engines", 0) > 0:
                for field in ("engines", "compute_units"):
                    require(begin[field] == row[field] and type(row[field]) is int and
                            row[field] > 0, f"invalid resource configuration: {field}")
                require(type(row["engine"]) is int and 0 <= row["engine"] < row["engines"],
                        "invalid engine id")
                require(all(cu == row["engine"] % row["compute_units"] for _, _, cu, _ in spans),
                        "compute stage escaped its engine CU")
                engines.setdefault(identity[:3] + (row["engine"],), []).append(
                    (stamp(row, "started")[1], stamp(row, "ready")[1]))
            for left, right, cu, _ in spans:
                occupancy.setdefault(identity[:3] + (cu,), []).append((left, right))
        else:
            raise ValueError("unknown event")
    require(not pending, f"{len(pending)} request(s) missing completion")
    require(bool(reports), "no complete v3 requests")
    for spans in occupancy.values():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            require(previous[1] <= current[0], "target CU double booked")
    for intervals in engines.values():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            require(previous[1] <= current[0], "autonomous engine double booked")
    return reports


def summarize_concurrency(records):
    groups = {}
    for row in records:
        if row["event"] != "complete":
            continue
        identity = tuple(row[name] for name in ("pid", "session", "controller"))
        events = groups.setdefault(identity, [])
        request = row["request"]

        def interval(kind, start, end):
            if end > start:
                events.extend(((start, 1, kind, request), (end, -1, kind, request)))

        interval("parent", stamp(row, "started")[1], stamp(row, "ready")[1])
        for phase in row["computes"]:
            for span in phase["cu_spans"]:
                interval("compute", span["start_ns"], span["end_ns"])
        for read in row["reads"]:
            interval("io", stamp(read, "submitted")[1], read["deadline_ns"])
    result = dict(max_active_requests=0, cross_request_compute_ns=0,
                  cross_request_io_compute_ns=0)
    for events in groups.values():
        active = {kind: {} for kind in ("parent", "compute", "io")}
        previous = 0
        for current, delta, kind, request in sorted(events):
            duration = current - previous
            if len(active["compute"]) > 1:
                result["cross_request_compute_ns"] += duration
            if any(reader != computer for reader in active["io"] for computer in active["compute"]):
                result["cross_request_io_compute_ns"] += duration
            counts = active[kind]
            counts[request] = counts.get(request, 0) + delta
            if not counts[request]:
                del counts[request]
            result["max_active_requests"] = max(result["max_active_requests"], len(active["parent"]))
            previous = current
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--read-summary", type=Path)
    parser.add_argument("--boundary-summary", type=Path)
    parser.add_argument("--require-read-diagnostics", action="store_true")
    parser.add_argument("--latest-session", action="store_true")
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--require-concurrent-compute", action="store_true")
    parser.add_argument("--require-cross-request-io", action="store_true")
    args = parser.parse_args()
    try:
        with args.trace.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        records = select_records(records, args.latest_session, args.min_tokens, args.max_tokens)
        reports = analyze(records)
        concurrency = summarize_concurrency(records)
        if args.require_concurrent_compute:
            require(concurrency["cross_request_compute_ns"] > 0,
                    "no observed cross-request CU overlap in virtual time")
        if args.require_cross_request_io:
            require(concurrency["cross_request_io_compute_ns"] > 0,
                    "no observed cross-request IO/CU overlap in virtual time")
        summary = summarize_reads(reports)
        boundaries = summarize_boundaries(reports)
        if args.require_read_diagnostics:
            require(all(not phase["missing_requests"] for phase in summary.values()),
                    "read diagnostics missing; enable CEMU_SPARF_READ_DIAGNOSTICS=1 and rerun")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"[sparf-autonomous] FAIL/INCOMPLETE: {error}\n")
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if args.read_summary:
        args.read_summary.write_text(json.dumps(summary, indent=2) + "\n")
    if args.boundary_summary:
        args.boundary_summary.write_text(json.dumps(boundaries, indent=2) + "\n")
    total_overlap = sum(row["qk_v_target_overlap_ns"] for row in reports)
    print(f"[sparf-autonomous] PASS: {len(reports)} requests; IO/CU exclusion and QK/V→PV dependencies valid")
    print(f"[sparf-autonomous] target QK/V overlap={total_overlap / 1e6:.6f} ms; "
          f"zero-budget phases={sum(row['zero_budget_phases'] for row in reports)}")
    print("[sparf-autonomous] Host intervals are not target times; PASS does not calibrate the timing model")
    print(f"[sparf-autonomous] concurrency={concurrency}")
    print(f"[sparf-boundary] parent Host sum={boundaries['host_elapsed_ns'] / 1e9:.6f}s, "
          f"virtual sum={boundaries['virtual_elapsed_ns'] / 1e9:.6f}s, "
          f"engine queue virtual sum={boundaries['engine_queue_virtual_ns'] / 1e9:.6f}s")
    for category in ("read_boundaries", "compute_boundaries"):
        for phase, values in boundaries[category].items():
            durations = ', '.join(f"{name}={value / 1e9:.6f}s" for name, value in values.items())
            print(f"[sparf-boundary] {phase}: {durations}")
    print("[sparf-boundary] Sums overlap; boundary intervals do not isolate freeze cost, "
          "GPU kernel time, or Guest ioctl completion latency")
    for phase, values in summary.items():
        if not values["observed_requests"]:
            print(f"[sparf-read] {phase}: diagnostics unavailable (not zero IO)")
            continue
        print(f"[sparf-read] {phase}: requests={values['observed_requests']}, missing={values['missing_requests']}, "
              f"logical={values['logical_bytes']} B, tail={values['tail_bytes']} B, "
              f"submitted={values['submitted_bytes']} B, pages={values['pages']}, "
              f"unmapped={values['unmapped_pages']}, dedup_saved={values['dedup_saved_bytes']} B")
        print(f"[sparf-read] {phase}: target_sum={values['model_ns'] / 1e9:.6f}s, "
              f"critical_initial_sum={values['critical_initial_wait_ns'] / 1e9:.6f}s, "
              f"critical_subsequent_sum={values['critical_subsequent_queue_ns'] / 1e9:.6f}s, "
              f"critical_service_sum={values['critical_service_ns'] / 1e9:.6f}s; "
              f"Host submit→observed={values['host_submit_to_observed_ns'] / 1e9:.6f}s, "
              f"observed→ready={values['host_observed_to_ready_ns'] / 1e9:.6f}s")
        print(f"[sparf-read] {phase}: active_luns_mean={values['active_luns_mean']}, "
              f"worst_lun_max/mean={values['lun_max_to_mean']}, "
              f"virtual_deadline_lateness_sum={values['virtual_deadline_to_ready_ns'] / 1e6:.6f}ms")
    print("[sparf-read] Phase/counter sums are not end-to-end latency; LUNs and QK/V may overlap")


if __name__ == "__main__":
    main()
