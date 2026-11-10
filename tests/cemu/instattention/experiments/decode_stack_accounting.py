from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from pwrite_stack_trace import interval_union, interval_size, interval_intersection, interval_subtract


CALLS = {
    "prefill.fdm_write.call_wall": ("prefill_append", "pwrite64", 5, "kv_buffer_data"),
    "decode.fdm_write.call_wall": ("kv_append", "pwrite64", 5, "kv_buffer_data"),
    "drain.fdm_write.call_wall": ("drain", "pwrite64", 5, "kv_buffer_data"),
    "attention.query.write.call_wall": ("query_write", "pwrite64", 5, "_query"),
    "attention.value_mean.write.call_wall": ("mean_write", "pwrite64", 5, "_value_mean"),
    "attention.output.read.call_wall": ("output_read", "pread64", 2, "_output"),
    "attention.trace.read.call_wall": ("trace_read", "pread64", 2, "_trace"),
    "attention.execute.call_wall": ("execute", "ioctl", 1, None),
}
REQUIRED = {"kv_append", "query_write", "mean_write", "output_read", "trace_read", "execute"}


def command_key(command):
    return (*command["key"], command["nsid"], command["opcode"], command["offset"], command["bytes"])


def shifted_bounds(spans, lower, upper):
    spans = interval_union(spans)
    return ([(start + lower, end + upper) for start, end in spans],
            [(start + upper, end + lower) for start, end in spans if start + upper < end + lower])


def analyze_decode_stack(trace, calls, devices, alignment):
    if (trace.get("version") != 1 or trace.get("request_context_version") != 1 or
            trace.get("clocksource") != "hpet" or trace.get("clock") != "clock_gettime(CLOCK_MONOTONIC)" or
            not trace.get("stack") or trace.get("dropped")):
        raise ValueError("Decode stack requires complete request-tagged HPET interval capture")
    lower, upper = alignment["offset_lower_ns"], alignment["offset_upper_ns"]
    tolerance = alignment.get("guest_trace_tolerance_ns", 2000)
    profile_tolerance = alignment.get("guest_profile_tolerance_ns", 1000)
    if lower > upper:
        raise ValueError("invalid clock alignment")
    vllm = any(entry["name"].startswith("vllm.") for entry in trace["intervals"])
    window_name = "vllm.decode.step_wall" if vllm else "framework.decode_iterations_wall"
    windows = interval_union((entry["start_ns"], entry["end_ns"]) for entry in trace["intervals"]
                             if entry["name"] == window_name)
    window_ns = interval_size(windows)
    if not window_ns:
        raise ValueError("missing Decode window")
    device_index = defaultdict(list)
    for index, record in enumerate(devices):
        if record.get("source") == "background_aggregate":
            continue
        if record.get("version") != 2 or "opcode" not in record:
            raise ValueError("device IO trace v2 required; rebuild QEMU and start a fresh VM")
        key = tuple(record[field] for field in ("controller", "qid", "cid", "nsid", "opcode", "offset", "bytes"))
        device_index[key].append((index, record))
    by_thread = defaultdict(list)
    for index, call in enumerate(calls):
        by_thread[call["tid"]].append((index, call))
    for entries in by_thread.values():
        entries.sort(key=lambda item: item[1]["enter"])
    starts = {tid: [call["enter"] for _, call in entries] for tid, entries in by_thread.items()}
    used_calls, used_device = set(), set()
    processes = set()
    compatible_lower, compatible_upper = lower, upper
    request_counts = defaultdict(Counter)
    request_coordinates = {}
    expected_counts = {}
    rows = []
    device_spans = []
    for entry in trace["intervals"]:
        if entry["name"] not in CALLS:
            continue
        if vllm and not interval_intersection(windows, [(entry["start_ns"], entry["end_ns"])]):
            continue
        kind, syscall, opcode, suffix = CALLS[entry["name"]]
        context = entry.get("request")
        if kind in REQUIRED:
            if context is None:
                raise ValueError(f"missing request identity: {kind}")
            identity = context["id"]
            expected = Counter({name: 1 for name in REQUIRED})
            if vllm:
                batch = context.get("rows", [])
                if (context.get("framework") != "vllm" or not batch or
                        type(context.get("read_trace")) is not bool or
                        any(row["prefill"] or row["count"] != 1 for row in batch) or
                        len({row["slot"] for row in batch}) != len(batch)):
                    raise ValueError("invalid pure-Decode vLLM batch identity")
                coordinates = (context["step"], context["layer"], context["read_trace"],
                    tuple((row["request_id"], row["slot"], row["start"], row["count"]) for row in batch))
                expected["kv_append"] = len(batch)
                if not context["read_trace"]:
                    del expected["trace_read"]
            else:
                coordinates = tuple(context[field] for field in ("layer", "microbatch", "tokens"))
            if request_coordinates.setdefault(identity, coordinates) != coordinates:
                raise ValueError("request coordinates changed within Decode")
            expected_counts[identity] = expected
            request_counts[identity][kind] += 1
        thread_starts = starts.get(entry["tid"], [])
        first = bisect_left(thread_starts, entry["start_ns"] - profile_tolerance)
        last = bisect_right(thread_starts, entry["end_ns"] + profile_tolerance)
        matches = [(index, call) for index, call in by_thread[entry["tid"]][first:last]
                   if call["exit"] <= entry["end_ns"] + profile_tolerance]
        if not matches:
            raise ValueError(f"missing syscall for {entry['name']}; pread/ioctl trace required")
        for index, call in matches:
            if index in used_calls or call["syscall"] != syscall or not call["commands"]:
                raise ValueError("ambiguous/wrong syscall association")
            used_calls.add(index)
            if suffix is not None and not call.get("filename", "").endswith(suffix):
                raise ValueError("client scope and FDM filename differ")
            if syscall != "ioctl":
                if (call["returned"] != call["bytes"] or
                        sum(command["bytes"] for command in call["commands"]) != call["bytes"]):
                    raise ValueError("short IO or incomplete split-command byte coverage")
            elif call["returned"] != 0 or len(call["commands"]) != 1:
                raise ValueError("failed or split EXEC ioctl")
            own_device = []
            command_details = []
            for command in call["commands"]:
                if command["coverage"] != 2 or command["opcode"] != opcode or command.get("status") != 0:
                    raise ValueError("incomplete/failed NVMe trace; rebuild and boot the coverage=2 kernel")
                boundaries = [call["enter"], command["prepared"], command["sq_copy"],
                              command["cq_seen"], command["complete"], call["exit"]]
                if boundaries != sorted(boundaries):
                    raise ValueError("non-monotonic Guest command boundaries")
                candidates = [(position, record) for position, record in device_index[command_key(command)]
                    if max(lower, command["sq_copy"] - record["virtual"]["sq_seen"] - tolerance) <=
                       min(upper, command["complete"] - record["virtual"]["cq_begin"] + tolerance)]
                if len(candidates) != 1 or candidates[0][0] in used_device:
                    raise ValueError(f"ambiguous/missing device command match: kind={kind} "
                                     f"key={command_key(command)} candidates={len(candidates)}; "
                                     "requires unique identity and compatible SQ..nvme_complete_rq bounds")
                position, record = candidates[0]
                used_device.add(position)
                processes.add((record["pid"], record["controller"]))
                if record["status"]:
                    raise ValueError("device IO failed")
                stamps = record["virtual"]
                compatible_lower = max(compatible_lower, command["sq_copy"] - stamps["sq_seen"] - tolerance)
                compatible_upper = min(compatible_upper, command["complete"] - stamps["cq_begin"] + tolerance)
                if compatible_lower > compatible_upper:
                    raise ValueError("no common Guest/device clock offset across all IO classes")
                if not 0 <= stamps["sq_seen"] <= stamps["cq_begin"] <= stamps["cq_end"]:
                    raise ValueError("invalid device completion boundaries")
                detail = dict(qid=record["qid"], cid=record["cid"], nsid=record["nsid"],
                              guest_sq_copy_ns=command["sq_copy"], guest_cq_seen_ns=command["cq_seen"],
                              guest_complete_ns=command["complete"], device_sq_seen_ns=stamps["sq_seen"],
                              device_cq_begin_ns=stamps["cq_begin"],
                              prepared_to_sq_ns=command["sq_copy"] - command["prepared"],
                              cq_seen_to_complete_ns=command["complete"] - command["cq_seen"])
                if opcode == 1:
                    boundaries = [stamps[field] for field in
                        ("sq_seen", "exec_enter", "exec_admitted", "exec_started", "exec_ready", "cq_begin")]
                    if boundaries != sorted(boundaries) or not stamps["exec_ready"]:
                        raise ValueError("incomplete autonomous EXEC device boundaries")
                    detail.update(device_admission_queue_ns=stamps["exec_started"] - stamps["exec_admitted"],
                                  device_execution_ns=stamps["exec_ready"] - stamps["exec_started"],
                                  device_ready_to_cq_ns=stamps["cq_begin"] - stamps["exec_ready"])
                else:
                    boundaries = [stamps[field] for field in ("sq_seen", "memory_enter", "locked", "dma_begin", "dma_end", "memory_done")]
                    if boundaries != sorted(boundaries) or max(stamps["model_ready"], stamps["memory_done"]) > stamps["cq_begin"]:
                        raise ValueError("incomplete FDM device boundaries")
                own_device.append((stamps["sq_seen"], stamps["cq_begin"]))
                command_details.append(detail)
            device_spans.extend(own_device)
            target = interval_intersection(windows, [(call["enter"], call["exit"])])
            rows.append(dict(kind=kind, request=context, tid=call["tid"], enter_ns=call["enter"], exit_ns=call["exit"],
                commands=command_details, syscall_decode_ns=interval_size(target), device_spans=own_device))
    if not request_counts or any(counts != expected_counts[identity] for identity, counts in request_counts.items()):
        raise ValueError("Decode IO coverage differs from expected append/Q/mean/EXEC/output/optional-trace counts")
    wait_name = "decode.batch_wait_wall" if vllm else "framework.attention_result_wait_wall"
    waited = [(entry.get("request") or {}).get("id") for entry in trace["intervals"]
              if entry["name"] == wait_name and (not vllm or
                  interval_intersection(windows, [(entry["start_ns"], entry["end_ns"])]))]
    if Counter(waited) != Counter({identity: 1 for identity in request_counts}):
        raise ValueError("Decode Future and IO request coverage differ")
    if len(processes) != 1:
        raise ValueError("mixed device processes/controllers")
    for index, call in enumerate(calls):
        if index not in used_calls and interval_intersection(windows, [(call["enter"], call["exit"])]):
            raise ValueError("unattributed FDM/EXEC syscall overlaps Decode")
    lower, upper = compatible_lower, compatible_upper
    alignment = dict(alignment, offset_lower_ns=lower, offset_upper_ns=upper, uncertainty_ns=upper - lower,
                     decode_completion_bound="nvme_complete_rq",
                     assumption="Unit-rate Guest HPET/QEMU virtual; append-derived seed refined with all SQ..nvme_complete_rq bounds; not a clock calibration")
    cq_anomalies = []
    for row in rows:
        target = interval_intersection(windows, [(row["enter_ns"], row["exit_ns"])])
        possible, guaranteed = shifted_bounds(row.pop("device_spans"), lower, upper)
        minimum, maximum = interval_subtract(target, possible), interval_subtract(target, guaranteed)
        row.update(software_min_spans=minimum, software_max_spans=maximum,
                   software_min_ns=interval_size(minimum), software_max_ns=interval_size(maximum))
        for command in row["commands"]:
            cq_lower = command["guest_sq_copy_ns"] - command["device_sq_seen_ns"] - tolerance
            cq_upper = command["guest_cq_seen_ns"] - command["device_cq_begin_ns"] + tolerance
            command["cq_seen_alignment_consistent"] = max(lower, cq_lower) <= min(upper, cq_upper)
            if not command["cq_seen_alignment_consistent"]:
                cq_anomalies.append(dict(kind=row["kind"], request=row["request"], tid=row["tid"], **command))
    minimum = interval_union(span for row in rows for span in row["software_min_spans"])
    maximum = interval_union(span for row in rows for span in row["software_max_spans"])
    possible, guaranteed = shifted_bounds(device_spans, lower, upper)
    uncovered_min = interval_subtract(minimum, possible)
    uncovered_max = interval_subtract(maximum, guaranteed)

    def metric(low, high):
        return dict(min_ns=interval_size(low), max_ns=interval_size(high),
                    fraction_min=interval_size(low) / window_ns, fraction_max=interval_size(high) / window_ns)

    by_kind = {}
    for kind in dict.fromkeys(row["kind"] for row in rows):
        selected = [row for row in rows if row["kind"] == kind and row["syscall_decode_ns"]]
        syscall_ns = interval_size(interval_intersection(windows,
            [(row["enter_ns"], row["exit_ns"]) for row in selected]))
        software = metric([span for row in selected for span in row["software_min_spans"]],
                          [span for row in selected for span in row["software_max_spans"]])
        by_kind[kind] = dict(calls=len(selected), summed_syscall_ns=sum(row["syscall_decode_ns"] for row in selected),
            syscall_union_ns=syscall_ns, software_union=software,
            syscall_outside_fraction_min=software["min_ns"] / syscall_ns if syscall_ns else None,
            syscall_outside_fraction_max=software["max_ns"] / syscall_ns if syscall_ns else None)
    report = dict(version=2, status="complete", decode_window_ns=window_ns, requests=len(request_counts),
        cq_seen_alignment_anomalies=cq_anomalies,
        clock_alignment=alignment, software_union=metric(minimum, maximum),
        software_not_overlapped_by_tracked_device=metric(uncovered_min, uncovered_max), by_kind=by_kind, rows=rows,
        notes=[
            "Scope: synchronous CEMU FDM pread/pwrite and autonomous EXEC ioctl, not all Python/FlexGen software.",
            "Device-owned envelope is SQ consume..CQ publication BEGIN; includes dispatch, device queues, model waits and poller delay.",
            "CQ publication and subsequent Guest notification/wakeup remain outside that envelope; cq_end is not a safe causal alignment boundary.",
            "Subtract each syscall's OWN device envelopes before taking the Decode-clipped union; other requests cannot hide its software path.",
            "Then optionally subtract all matched foreground-device envelopes to describe overlap; background NAND not covered.",
            "Includes blocking/transport/scheduling and instrumentation time, not pure CPU cost or a guaranteed optimization gain.",
            "All commands use SQ..nvme_complete_rq for causal matching; CQ-seen conflicts are reported, never dropped or used to force alignment.",
            "Completion-bound consistency does not prove every timestamp accurate; reported ranges cover offset uncertainty, not arbitrary clock sampling error.",
            "Per-kind unions can overlap and cannot be added. Clock-offset bounds are not a calibration of HPET or hardware.",
        ])
    if vllm:
        report["framework"] = "vllm"
        report["broad_stack"] = vllm_stack_breakdown(trace, calls, windows, possible, guaranteed,
                                                    minimum, maximum)
    return report


def vllm_stack_breakdown(trace, calls, windows, possible, guaranteed, minimum, maximum):
    named = defaultdict(list)
    for entry in trace["intervals"]:
        named[entry["name"]].append((entry["start_ns"], entry["end_ns"]))
    diagnostics = named["decode.trace_read_wall"]
    clean = interval_subtract(windows, diagnostics)
    denominator = interval_size(clean)
    if not denominator:
        raise ValueError("no Decode interval remains after excluding diagnostic reads")
    syscalls = [(call["enter"], call["exit"]) for call in calls]
    client_calls = [span for name, spans in named.items() if name.endswith(".call_wall") for span in spans]
    preparation = [span for name in ("decode.append_wall", "decode.value_sum_wall",
        "decode.mean_and_rows_wall", "decode.metadata_wall") for span in named[name]]
    preparation = interval_subtract(preparation, client_calls)
    wrappers = interval_subtract(client_calls, syscalls)
    outer_calls = [span for name in ("decode.batch_wait_wall", "decode.fdm_write_wall",
                                    "decode.output_read_wall") for span in named[name]]
    wrappers += interval_subtract(outer_calls, client_calls + preparation)
    transfers = [span for name in ("decode.d2h_enqueue_wall", "decode.d2h_wait_wall",
                                  "decode.h2d_enqueue_wall") for span in named[name]]

    def metric(low, high):
        low_ns = interval_size(interval_intersection(clean, low))
        high_ns = interval_size(interval_intersection(clean, high))
        return dict(min_ns=low_ns, max_ns=high_ns,
                    fraction_min=low_ns / denominator, fraction_max=high_ns / denominator)

    upper_scopes = preparation + wrappers + transfers
    broad_min = interval_union(minimum + upper_scopes)
    broad_max = interval_union(maximum + upper_scopes)
    return dict(decode_without_trace_ns=denominator,
        diagnostic_excluded_ns=interval_size(windows) - denominator,
        syscall_outside_own_device=metric(minimum, maximum),
        preparation=metric(preparation, preparation), client_wrappers=metric(wrappers, wrappers),
        transfer_api_and_wait=metric(transfers, transfers),
        broad_software_union=metric(broad_min, broad_max),
        foreground_device_union=metric(guaranteed, possible),
        software_device_overlap=metric(interval_intersection(broad_min, guaranteed),
                                      interval_intersection(broad_max, possible)),
        remaining=metric(interval_subtract(clean, broad_max + possible),
                         interval_subtract(clean, broad_min + guaranteed)),
        notes=[
            "Denominator is actual pure-Decode worker execute/sample windows, excluding trace read scopes; mixed steps are separate.",
            "Broad software includes caller-visible GPU/CPU transfer enqueue/wait, preparation, mean, binding wrappers and syscall complements.",
            "Transfer waits can contain producer GPU work; these are elapsed API intervals, not exclusive DMA or CPU costs.",
            "Foreground device union is matched SQ-consume..CQ-publication-begin, including queues and aggregation dependency waits.",
            "Background work outside foreground envelopes is not attributed; remaining includes model GPU compute, framework/sampling and gaps.",
            "Broad software and device may overlap; their union, not their sum, covers the denominator. This is not an optimization upper bound.",
        ])
