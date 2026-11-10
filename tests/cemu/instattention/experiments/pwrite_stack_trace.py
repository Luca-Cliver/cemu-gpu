import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import threading
import uuid


EVENTS = (
    "syscalls/sys_enter_pwrite64", "syscalls/sys_exit_pwrite64",
    "syscalls/sys_enter_pread64", "syscalls/sys_exit_pread64",
    "syscalls/sys_enter_ioctl", "syscalls/sys_exit_ioctl",
    "fdmfs/fdmfs_rw_begin", "fdmfs/fdmfs_rw_end",
    "iomap/iomap_dio_rw_begin", "iomap/iomap_dio_complete",
    "nvme/nvme_cemu_io", "nvme/nvme_complete_rq",
)
AGGREGATE_STAGES = ("validate", "preflight", "free_list", "token_k", "channel_k", "token_v", "commit")
LINE = re.compile(r"-(\d+)\s+(?:\(\s*\d+\)\s+)?\[\d+\]\s+\S+\s+(\d+)\.(\d+):\s+(.*)")
FIELDS = re.compile(r"(\w+)=([^\s,]+)")


def decode_line(line):
    if "LOST" in line or "EVENTS DROPPED" in line:
        raise ValueError("kernel trace lost events")
    match = LINE.search(line)
    if not match:
        return None
    tid, seconds, fraction, payload = match.groups()
    stamp = int(seconds) * 10**9 + int(fraction.ljust(9, "0")[:9])
    syscall = re.match(r"(?:sys_)?(pwrite64|pread64|ioctl)(\(| ->)", payload)
    if syscall:
        event = ("sys_enter_" if syscall[2] == "(" else "sys_exit_") + syscall[1]
        body = payload
    else:
        parts = payload.split(": ", 1)
        if len(parts) != 2:
            return None
        event, body = parts
        event = event.split(":")[-1]
    return int(tid), stamp, event, body


def parse_guest(lines, filename="kv_buffer_data", all_io=False):
    active = {}
    requests = {}
    completed = []
    for line in lines:
        event = decode_line(line)
        if event is None:
            continue
        tid, stamp, name, body = event
        fields = dict(FIELDS.findall(body))
        if name == "sys_enter_pwrite64" or all_io and name in ("sys_enter_pread64", "sys_enter_ioctl"):
            if tid in active:
                raise ValueError("nested/missing pwrite exit")
            active[tid] = dict(tid=tid, enter=stamp, commands=[], syscall=name.removeprefix("sys_enter_"))
            continue
        call = active.get(tid)
        if name == "fdmfs_rw_begin" and call is not None:
            if all_io or fields.get("name") == filename:
                call.update(fdm=stamp, bytes=int(fields["bytes"]), offset=int(fields["offset"]), filename=fields.get("name"))
        elif name in ("iomap_dio_rw_begin", "iomap_dio_complete", "fdmfs_rw_end"):
            if call is not None and "fdm" in call:
                stage = {"iomap_dio_rw_begin": "iomap", "iomap_dio_complete": "dio_done",
                         "fdmfs_rw_end": "fdm_done"}[name]
                if stage in call:
                    raise ValueError(f"duplicate {stage} for pwrite")
                call[stage] = stamp
        elif name == "nvme_cemu_io":
            key = tuple(int(fields[field]) for field in ("ctrl", "qid", "cid"))
            stage = fields["stage"]
            if stage == "prepared":
                execute = all_io and int(fields.get("opcode", "0")) == 1 and call is not None and call["syscall"] == "ioctl"
                if call is None or "fdm" not in call and not execute:
                    continue
                if key in requests:
                    raise ValueError("NVMe command ID reused before completion")
                command = dict(key=key, prepared=stamp, nsid=int(fields["nsid"]),
                               bytes=int(fields["bytes"]), offset=int(fields["offset"]),
                               opcode=int(fields.get("opcode", "5")), coverage=int(fields.get("coverage", "1")))
                call["commands"].append(command)
                requests[key] = command
            elif key in requests:
                if stage in requests[key]:
                    raise ValueError(f"duplicate NVMe {stage}")
                requests[key][stage] = stamp
        elif name == "nvme_complete_rq":
            controller = re.search(r"nvme(\d+):", body)
            if controller and "qid" in fields and "cmdid" in fields:
                key = (int(controller[1]), int(fields["qid"]), int(fields["cmdid"]))
                command = requests.pop(key, None)
                if command is not None:
                    command["complete"] = stamp
                    command["status"] = int(fields["status"], 0)
        elif call is not None and name == "sys_exit_" + call["syscall"]:
            active.pop(tid)
            if "fdm" in call or call["commands"]:
                result = re.search(r"(?:->|ret:)\s*(0x[0-9a-fA-F]+|-?\d+)", body)
                if result is None:
                    result = re.fullmatch(r"\s*(0x[0-9a-fA-F]+|-?\d+)\s*", body)
                if result is None:
                    raise ValueError("cannot parse pwrite return value")
                call.update(exit=stamp, returned=int(result[1], 0))
                completed.append(call)
    if requests or any("fdm" in call or call["commands"] for call in active.values()):
        raise ValueError("unfinished target pwrite/NVMe requests")
    return completed


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0, total_seconds=0, average_us=0, maximum_us=0, p95_us=0, p99_us=0)
    return dict(count=len(ordered), total_seconds=sum(ordered) / 1e9,
                average_us=sum(ordered) / len(ordered) / 1000, maximum_us=ordered[-1] / 1000,
                p95_us=ordered[math.ceil(len(ordered) * .95) - 1] / 1000,
                p99_us=ordered[math.ceil(len(ordered) * .99) - 1] / 1000)


def difference(end, start):
    if end < start:
        raise ValueError("non-monotonic/mismatched trace boundaries")
    return end - start


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        difference(end, start)
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def interval_size(intervals):
    return sum(end - start for start, end in interval_union(intervals))


def interval_intersection(left, right):
    left, right = interval_union(left), interval_union(right)
    result = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append((start, end))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def interval_subtract(left, right):
    right = interval_union(right)
    result = []
    right_index = 0
    for start, end in interval_union(left):
        while right_index < len(right) and right[right_index][1] <= start:
            right_index += 1
        index = right_index
        while index < len(right) and right[index][0] < end:
            other_start, other_end = right[index]
            if start < other_start:
                result.append((start, other_start))
            start = max(start, other_end)
            if start >= end:
                break
            index += 1
        if start < end:
            result.append((start, end))
    return result


def outside_device(target, device, alignment):
    lower, upper = alignment["offset_lower_ns"], alignment["offset_upper_ns"]
    device = interval_union(device)
    possible = [(start + lower, end + upper) for start, end in device]
    guaranteed = [(start + upper, end + lower) for start, end in device if start + upper < end + lower]
    total = interval_size(target)
    minimum = total - interval_size(interval_intersection(target, possible))
    maximum = total - interval_size(interval_intersection(target, guaranteed))
    if not interval_subtract(possible, target):
        minimum = maximum = total - interval_size(device)
    return dict(target_ns=total, outside_min_ns=minimum, outside_max_ns=maximum,
                outside_fraction_min=minimum / total if total else None,
                outside_fraction_max=maximum / total if total else None)


def device_accounting(pairs, phase_trace=None, attention_records=()):
    tolerance = 2000
    lower = upper = None
    processes = set()
    sessions = set()
    vllm = phase_trace is not None and any(
        entry["name"].startswith("vllm.") for entry in phase_trace["intervals"])
    for call, row, matches in pairs:
        for command, record in matches:
            stamps = record["virtual"]
            processes.add((record.get("pid"), record["controller"]))
            sessions.add(record["session"])
            minimum = command["sq_copy"] - stamps["sq_seen"] - tolerance
            maximum = command["complete" if vllm else "cq_seen"] - stamps["cq_begin"] + tolerance
            lower = minimum if lower is None else max(lower, minimum)
            upper = maximum if upper is None else min(upper, maximum)
    if lower is None or lower > upper or len(processes) != 1 or len(sessions) != 1:
        raise ValueError("no consistent unit-rate Guest/virtual clock offset; mixed runs, drift or incomplete trace")
    alignment = dict(offset_lower_ns=lower, offset_upper_ns=upper, uncertainty_ns=upper - lower,
                     guest_trace_tolerance_ns=tolerance,
                     assumption="Guest HPET and QEMU virtual have equal rate; causal SQ/CQ bounds, not a clock calibration")
    if vllm:
        alignment["assumption"] = "Unit-rate Guest HPET/QEMU virtual; causal SQ..nvme_complete_rq bounds, not a clock calibration"
    all_service = []
    by_phase = defaultdict(list)
    for call, row, matches in pairs:
        service = []
        envelopes = []
        for command, record in matches:
            stamps = record["virtual"]
            ready = max(stamps["memory_done"], stamps["model_ready"])
            envelopes.append((stamps["sq_seen"], stamps["cq_begin"]))
            service.append((stamps["memory_enter"], ready))
            if record["checks"]:
                service.append((stamps["first_check"], stamps["last_check"]))
        target = [(call["enter"], call["exit"])]
        row["device_accounting"] = dict(
            service_and_wait=outside_device(target, service, alignment),
            request_envelope=outside_device(target, envelopes, alignment),
            service_union_ns=interval_size(service),
            command_service_sum_ns=sum(end - start for start, end in service))
        all_service.extend(service)
        by_phase[row["phase"]].append(row["device_accounting"])
    summary = {}
    for phase, entries in by_phase.items():
        summary[phase] = {}
        for boundary in ("service_and_wait", "request_envelope"):
            totals = {field: sum(entry[boundary][field] for entry in entries)
                      for field in ("target_ns", "outside_min_ns", "outside_max_ns")}
            totals.update(outside_fraction_min=totals["outside_min_ns"] / totals["target_ns"] if totals["target_ns"] else None,
                          outside_fraction_max=totals["outside_max_ns"] / totals["target_ns"] if totals["target_ns"] else None)
            summary[phase][boundary] = totals
    report = dict(clock_alignment=alignment, syscall_by_phase=summary, notes=[
        "Fractions use summed syscall durations, not mean per-call percentages; concurrent calls may overlap.",
        "service_and_wait = memory_enter..max(memory_done,model_ready) plus first_check..last_check.",
        "The check span includes resource retry AND poller/aggregation-yield gaps; it is not pure device execution.",
        "request_envelope = sq_seen..cq_begin; CQ publication is deliberately excluded (cq_end may follow Guest observation).",
        "Outside intervals include protocol/scheduler waiting, not just CPU work; software can overlap device service.",
        "Bounds account for offset uncertainty; no Host monotonic durations are subtracted from Guest time.",
    ])
    if phase_trace is not None:
        report["pipeline"] = phase_accounting(phase_trace, pairs, all_service,
                                              alignment, attention_records, next(iter(processes))[0])
    return report


def decode_wait_accounting(trace, pairs, completed, alignment):
    if trace.get("request_context_version") != 1:
        return dict(status="unavailable", reason="rerun with request-correlated profiling")
    if not completed:
        return dict(status="unavailable", reason="Attention device timeline required")
    entries = defaultdict(lambda: defaultdict(list))
    contexts = {}
    executions = {}
    windows = []
    for entry in trace["intervals"]:
        span = (entry["start_ns"], entry["end_ns"])
        if entry["name"] == "framework.decode_iterations_wall":
            windows.append(span)
        context = entry.get("request")
        if context is None:
            if entry["name"] in ("framework.attention_result_wait_wall", "decode.execute_wait_wall"):
                raise ValueError("missing Decode request context")
            continue
        identity = context["id"]
        coordinates = {key: context[key] for key in ("layer", "microbatch", "tokens")}
        if contexts.setdefault(identity, coordinates) != coordinates:
            raise ValueError("Decode request identity reused for different coordinates")
        entries[identity][entry["name"]].append(span)
        if entry["name"] == "decode.execute_wait_wall":
            if identity in executions:
                raise ValueError("multiple EXEC intervals for one Decode request")
            executions[identity] = entry
    append_service = defaultdict(list)
    append_syscalls = defaultdict(list)
    for call, row, matches in pairs:
        if row["phase"] != "decode":
            continue
        context = row.get("request")
        if context is None or context["id"] not in contexts:
            raise ValueError("Decode append is missing a known request identity")
        identity = context["id"]
        if any(context[key] != contexts[identity][key] for key in contexts[identity]):
            raise ValueError("Decode append request coordinates differ")
        append_syscalls[identity].append((call["enter"], call["exit"]))
        for _, record in matches:
            stamps = record["virtual"]
            append_service[identity].append((stamps["memory_enter"], max(stamps["memory_done"], stamps["model_ready"])))
            if record["checks"]:
                append_service[identity].append((stamps["first_check"], stamps["last_check"]))
    lower, upper = alignment["offset_lower_ns"], alignment["offset_upper_ns"]
    by_execution = defaultdict(list)
    for index, record in enumerate(completed):
        by_execution[(record["program"], record["layer"], record["tokens"])].append((index, record))
    matched = set()
    rows = []
    all_waits = []
    for identity, named in entries.items():
        if identity not in executions or not append_syscalls[identity]:
            raise ValueError("incomplete Decode EXEC/append coverage")
        for required in ("framework.attention_result_wait_wall", "decode.attention.worker_wall",
                         "decode.attention.queue_wait_wall"):
            if len(named[required]) != 1:
                raise ValueError(f"incomplete or duplicate Decode boundary: {required}")
        execution = executions[identity]
        context = execution["request"]
        candidates = [index for index, record in by_execution[(context["program"], context["layer"], context["tokens"])]
            if max(lower, execution["start_ns"] - record["started"]["virtual_ns"]) <=
            min(upper, execution["end_ns"] - record["ready"]["virtual_ns"])]
        if len(candidates) != 1 or candidates[0] in matched:
            raise ValueError("ambiguous/missing request-specific Attention match")
        matched.add(candidates[0])
        device = completed[candidates[0]]
        service = interval_union(append_service[identity] + [
            (device["started"]["virtual_ns"], device["ready"]["virtual_ns"])])
        possible = [(start + lower, end + upper) for start, end in service]
        guaranteed = [(start + upper, end + lower) for start, end in service if start + upper < end + lower]
        waits = interval_intersection(windows, named["framework.attention_result_wait_wall"])
        all_waits.extend(waits)
        categories = [
            ("lock_wait", [span for name, spans in named.items() if name.endswith(".lock_wait_wall") for span in spans]),
            ("worker_queue", [span for name, spans in named.items() if name.endswith(".queue_wait_wall") for span in spans]),
            ("producer_or_prefill_wait", [span for name in ("decode.layer_prefill_wait_wall",
                "decode.producer_wait_wall", "framework.prefill_accept_wait_wall") for span in named[name]]),
            ("slot_wait", named["decode.slot_wait_wall"]),
            ("append_syscall_outside_device", append_syscalls[identity]),
            ("exec_transport_or_device_queue", named["decode.execute_wait_wall"]),
            ("query_result_trace_io", [span for name in ("decode.fdm_write_wall", "decode.output_read_wall",
                "decode.trace_read_wall") for span in named[name]]),
            ("kv_query_export", named["decode.kv_export_wall"] + named["decode.query_export_wall"]),
            ("packet_packing", named["decode.kv_payload_pack_wall"] + named["decode.fdm_packet_pack_wall"]),
            ("mean_and_metadata", [span for name in ("decode.value_mean_update_wall",
                "decode.value_mean_prepare_wall", "decode.metadata_wall") for span in named[name]]),
            ("append_binding_outside_syscall", named["decode.fdm_write.call_wall"]),
            ("return_to_waiter", [(named["decode.attention.worker_wall"][0][1],
                max(named["decode.attention.worker_wall"][0][1], named["framework.attention_result_wait_wall"][0][1]))]),
        ]
        bounds = []
        for coverage in (possible, guaranteed):
            remaining = interval_subtract(waits, coverage)
            partition = {}
            for name, spans in categories:
                part = interval_intersection(remaining, spans)
                partition[name] = interval_size(part)
                remaining = interval_subtract(remaining, part)
            partition["unclassified"] = interval_size(remaining)
            bounds.append(partition)
        rows.append(dict(request=identity, coordinates=contexts[identity], device_request=device["request"],
            wait_ns=interval_size(waits), outside_min_ns=sum(bounds[0].values()),
            outside_max_ns=sum(bounds[1].values()), categories={name: dict(min_ns=bounds[0][name], max_ns=bounds[1][name])
                for name in bounds[0]}))
    if len(matched) != len(completed):
        raise ValueError("unmatched Attention device requests")
    if interval_size(all_waits) != sum(row["wait_ns"] for row in rows):
        raise ValueError("overlapping main-thread wait intervals")
    totals = {key: sum(row[key] for row in rows) for key in ("wait_ns", "outside_min_ns", "outside_max_ns")}
    totals["categories"] = {name: {bound: sum(row["categories"][name][bound] for row in rows)
        for bound in ("min_ns", "max_ns")} for name in rows[0]["categories"]} if rows else {}
    return dict(status="complete", requests=len(rows), totals=totals, rows=rows, notes=[
        "Each Future wait is clipped to Decode and matched to ONLY its own append and Attention device intervals.",
        "Program/layer/token identity plus causal clock bounds must select exactly one Attention request; ambiguity fails analysis.",
        "Categories partition the uncovered wait with priority in listed order; nested scopes are not added together.",
        "Bounds are conservative under clock-offset uncertainty, not a point estimate of CPU or critical-path cost.",
        "Query/result/trace IO still includes untracked device work; EXEC residual includes device admission/notification waits.",
        "Return-to-waiter starts at callable-body exit, not an instrumented Future state transition; it includes scheduling/profiler work.",
        "Unclassified includes uninstrumented framework work and scheduling gaps; it must not be labelled pure software overhead.",
    ])


def phase_accounting(trace, pairs, service, alignment, attention_records, device_pid):
    if (trace.get("version") != 1 or trace.get("clock") != "clock_gettime(CLOCK_MONOTONIC)" or
            trace.get("clocksource") != "hpet" or not trace.get("stack") or trace.get("dropped", 0)):
        raise ValueError("phase intervals require stack profiling and Guest CLOCK_MONOTONIC/HPET")
    named = defaultdict(list)
    for entry in trace["intervals"]:
        named[entry["name"]].append((entry["start_ns"], entry["end_ns"]))
    for intervals in named.values():
        interval_union(intervals)
    vllm = any(name.startswith("vllm.") for name in named)
    io_calls = [span for name, spans in named.items() if name.endswith(".call_wall") for span in spans]
    preparation = [span for name, spans in named.items()
                   if name.endswith((".store_append_wall", ".metadata_wall", ".value_mean_prepare_wall",
                                     ".append_wall", ".mean_and_rows_wall", ".value_sum_wall"))
                   for span in spans]
    preparation = interval_subtract(preparation, io_calls)
    syscalls = [(call["enter"], call["exit"]) for call, _, _ in pairs]
    append_calls = [span for name, spans in named.items() if name in
                    ("prefill.fdm_write.call_wall", "decode.fdm_write.call_wall") for span in spans]
    preparation += interval_subtract(append_calls, syscalls)
    waits = [span for name in ("prefill.writer_backpressure_wall", "framework.attention_result_wait_wall")
             for span in named[name]]
    attention = []
    completed = []
    if attention_records:
        from analyze_sparf_autonomous import select_records, analyze as analyze_attention
        selected = select_records(attention_records, latest_session=True)
        analyze_attention(selected)
        completed = [record for record in selected if record.get("event") == "complete"]
        expected = len(named["decode.execute_wait_wall"])
        if not expected or len(completed) != expected or any(record["pid"] != device_pid for record in completed):
            raise ValueError("Attention timeline PID/count differs from this profiled run")
        for record in completed:
            attention.append((record["started"]["virtual_ns"], record["ready"]["virtual_ns"]))
        generation = named["pipeline.generate_wall"]
        if vllm:
            steps = [span for phase in ("prefill", "decode", "mixed")
                     for span in named[f"vllm.{phase}.step_wall"]]
            generation = [(min(start for start, _ in steps), max(end for _, end in steps))] if steps else []
        if len(generation) != 1 or any(
                start + alignment["offset_upper_ns"] < generation[0][0] or
                end + alignment["offset_lower_ns"] > generation[0][1] for start, end in attention):
            raise ValueError("Attention timeline is outside this generation interval")
        service = service + attention
    result = dict(attention_requests=len(attention), device_coverage="append service/wait spans" +
                  (" + Attention started..ready envelopes" if attention else "; Attention device coverage MISSING"),
                  software_stack_fraction=None, phases={}, notes=[
        "Phase windows come from actual FlexGen iteration start/stop pairs, not reconstructed from totals.",
        "All unions are clipped to phase windows, including Prefill writes spilling into Decode.",
        "Device complement is NOT software overhead: it includes GPU, framework work and untraced activity.",
        "Known client preparation excludes IO call envelopes; append binding overhead excludes pwrite syscalls.",
        "Blocking-wait complement is not a causal critical-path attribution and can include producer/scheduler stalls.",
        "No complete software-stack fraction is claimed: query/result device IO and background NAND are not covered here.",
        "Category unions can overlap each other; per-scope totals are inclusive, not additive or CPU time.",
    ])
    phase_names = (("prefill", "framework.prefill_iteration_wall"),
                   ("decode", "framework.decode_iterations_wall"),
                   ("drain", "pipeline.final_flush_wall"))
    if vllm:
        phase_names = tuple((phase, f"vllm.{phase}.step_wall") for phase in ("prefill", "decode", "mixed")) + (
            ("drain", "terminal_drain.wall"),)
        result["notes"][0] = "Phase windows are actual vLLM worker execute/sample intervals; frontend scheduling gaps are excluded."
    for phase, name in phase_names:
        windows = interval_union(named[name])
        if not windows:
            if vllm:
                continue
            raise ValueError(f"missing phase windows: {name}")
        window_ns = interval_size(windows)
        clipped_syscalls = interval_intersection(windows, syscalls)
        clipped_preparation = interval_intersection(windows, preparation)
        clipped_waits = interval_intersection(windows, waits)
        groups = dict(append_syscalls=clipped_syscalls, client_preparation=clipped_preparation,
                      main_thread_blocking_wait=clipped_waits)
        items = {}
        for group, spans in groups.items():
            item = outside_device(spans, service, alignment)
            item.update(phase_fraction_min=item["outside_min_ns"] / window_ns,
                        phase_fraction_max=item["outside_max_ns"] / window_ns)
            items[group] = item
        scope_totals = {}
        for scope, spans in named.items():
            clipped = interval_intersection(windows, spans)
            if clipped:
                scope_totals[scope] = dict(union_ns=interval_size(clipped),
                    inclusive_sum_ns=sum(interval_size(interval_intersection(windows, [span])) for span in spans))
        result["phases"][phase] = dict(window_ns=window_ns,
            foreground_device_complement=outside_device(windows, service, alignment),
            categories=items, scopes=scope_totals)
    result["decode_request_wait"] = (dict(status="not_applicable", reason="vLLM uses synchronous batch Attention; see decode_stack")
                                      if vllm else decode_wait_accounting(trace, pairs, completed, alignment))
    return result


def analyze(calls, packets, device_records=(), account_device=False, phase_trace=None, attention_records=()):
    samples = defaultdict(list)
    counters = defaultdict(int)
    rows = []
    used = set()
    by_tid = defaultdict(list)
    device = defaultdict(list)
    pairs = []
    for index, call in enumerate(calls):
        by_tid[call["tid"]].append((index, call))
    starts = {tid: [call["enter"] for _, call in entries] for tid, entries in by_tid.items()}
    for record in device_records:
        if record.get("input"):
            device[(record["session"], record["packet_id"])].append(record)
    for packet in packets:
        if not packet["success"]:
            raise ValueError("failed packet: timings cannot be used as successful IO")
        timestamps = starts.get(packet["tid"], [])
        first = bisect_left(timestamps, packet["start_ns"] - 1000)
        last = bisect_right(timestamps, packet["end_ns"] + 1000)
        matches = [(index, call) for index, call in by_tid[packet["tid"]][first:last]
                   if call["exit"] <= packet["end_ns"] + 1000]
        if not matches or sum(call["returned"] for _, call in matches) != packet["bytes"]:
            raise ValueError(f"packet {packet['packet_id']}: missing/split/failed pwrite coverage")
        phase = packet["phase"]
        counters[f"{phase}.packets"] += 1
        counters[f"{phase}.wire_bytes"] += packet["bytes"]
        commands = []
        for index, call in matches:
            if index in used:
                raise ValueError("ambiguous packet/pwrite association")
            used.add(index)
            if call["returned"] != call["bytes"] or not call["commands"]:
                raise ValueError("short write or missing NVMe request association")
            if sum(command["bytes"] for command in call["commands"]) != call["bytes"]:
                raise ValueError("NVMe byte coverage differs from pwrite; split/requeue trace incomplete")
            commands.extend(call["commands"])
            for command in call["commands"]:
                if command.get("status") != 0:
                    raise ValueError("NVMe request failed or completion missing")
                for name, start, end in (("request.prepared_to_sq", "prepared", "sq_copy"),
                                         ("request.sq_to_cq", "sq_copy", "cq_seen"),
                                         ("request.cq_to_complete", "cq_seen", "complete")):
                    samples[f"{phase}.{name}"].append(difference(command[end], command[start]))
            samples[f"{phase}.syscall_wall"].append(difference(call["exit"], call["enter"]))
            row = dict(packet_id=packet["packet_id"], phase=phase, tid=call["tid"],
                       bytes=call["bytes"], commands=len(call["commands"]),
                       enter_ns=call["enter"], exit_ns=call["exit"], request=packet.get("request"))
            roundtrips = [(command["sq_copy"], command["cq_seen"]) for command in call["commands"]]
            syscall = [(call["enter"], call["exit"])]
            if interval_subtract(roundtrips, syscall):
                raise ValueError("NVMe roundtrip is outside its pwrite syscall")
            row["guest_roundtrip_union_ns"] = interval_size(roundtrips)
            row["guest_outside_pending_nvme_ns"] = interval_size(interval_subtract(syscall, roundtrips))
            samples[f"{phase}.guest_roundtrip_union"].append(row["guest_roundtrip_union_ns"])
            samples[f"{phase}.guest_outside_pending_nvme"].append(row["guest_outside_pending_nvme_ns"])
            if len(call["commands"]) == 1:
                command = call["commands"][0]
                boundaries = [
                    ("vfs_entry", call["enter"], call["fdm"]),
                    ("fdm_entry", call["fdm"], call["iomap"]),
                    ("dio_block_driver", call["iomap"], command["prepared"]),
                    ("sq_submission", command["prepared"], command["sq_copy"]),
                    ("sq_cq_roundtrip", command["sq_copy"], command["cq_seen"]),
                    ("nvme_completion", command["cq_seen"], command["complete"]),
                    ("bio_wakeup_dio", command["complete"], call["dio_done"]),
                    ("fdm_return", call["dio_done"], call["fdm_done"]),
                    ("vfs_return", call["fdm_done"], call["exit"]),
                ]
                row["exclusive_guest_ns"] = {}
                for name, start, end in boundaries:
                    elapsed = difference(end, start)
                    samples[f"{phase}.single.{name}"].append(elapsed)
                    row["exclusive_guest_ns"][name] = elapsed
                if sum(row["exclusive_guest_ns"].values()) != call["exit"] - call["enter"]:
                    raise ValueError("exclusive Guest intervals do not sum to syscall duration")
            rows.append(row)
        if device_records:
            matches_device = device.pop((packet["session"], packet["packet_id"]), [])
            guest_keys = sorted((command["key"][1], command["key"][2], command["nsid"],
                                 command["offset"], command["bytes"]) for command in commands)
            device_keys = sorted((record["qid"], record["cid"], record["nsid"],
                                  record["offset"], record["bytes"]) for record in matches_device)
            if guest_keys != device_keys or len({record["controller"] for record in matches_device}) != 1:
                raise ValueError("device/Guest command identity or coverage mismatch")
            if account_device:
                lookup = {(record["qid"], record["cid"], record["nsid"], record["offset"], record["bytes"]): record
                          for record in matches_device}
                if len(lookup) != len(matches_device):
                    raise ValueError("ambiguous device command identity within packet")
                for (_, call), row in zip(matches, rows[-len(matches):]):
                    pairs.append((call, row, [(command, lookup[(command["key"][1], command["key"][2],
                        command["nsid"], command["offset"], command["bytes"])]) for command in call["commands"]]))
            for record in matches_device:
                if record["status"]:
                    raise ValueError("device request failed")
                stamps = record["virtual"]
                for field in ("checks", "aggregate_calls", "space_retries", "busy_retries", "cq_full"):
                    counters[f"{phase}.device.{field}"] += record[field]
                ready = max(stamps["memory_done"], stamps["model_ready"])
                intervals = [
                    ("dispatch", stamps["sq_seen"], stamps["memory_enter"]),
                    ("memory_lock", stamps["memory_enter"], stamps["locked"]),
                    ("receive_check", stamps["locked"], stamps["dma_begin"]),
                    ("dma_copy", stamps["dma_begin"], stamps["dma_end"]),
                    ("receive_finish", stamps["dma_end"], stamps["memory_done"]),
                    ("remaining_model_gate", stamps["memory_done"], ready),
                    ("cq_publish", stamps["cq_begin"], stamps["cq_end"]),
                ]
                if record["checks"]:
                    intervals.extend([
                        ("ready_to_check", ready, stamps["first_check"]),
                        ("check_span", stamps["first_check"], stamps["last_check"]),
                        ("check_to_cq", stamps["last_check"], stamps["cq_begin"]),
                    ])
                    gap = difference(stamps["last_check"], stamps["first_check"]) - record["check_ns"]
                    if gap < 0:
                        raise ValueError("invalid completion-check accounting")
                    samples[f"{phase}.device.retry_gap"].append(gap)
                    retry_fields = ("space_retry_ns", "busy_retry_ns", "drain_retry_ns", "model_retry_ns")
                    if "aggregate_yield_ns" in record:
                        retry_fields += ("aggregate_yield_ns",)
                    if sum(record[field] for field in retry_fields) != gap:
                        raise ValueError("retry reason intervals do not sum to retry gap")
                    for field in retry_fields:
                        samples[f"{phase}.device.{field}"].append(record[field])
                else:
                    intervals.append(("ready_to_cq", ready, stamps["cq_begin"]))
                for name, start, end in intervals:
                    samples[f"{phase}.device.{name}"].append(difference(end, start))
                for field in ("check_ns", "aggregate_ns"):
                    samples[f"{phase}.device.{field}"].append(record[field])
                samples[f"{phase}.device.check_other_ns"].append(
                    difference(record["check_ns"], record["aggregate_ns"]))
                samples[f"{phase}.host.aggregate_ns"].append(record["aggregate_host_ns"])
                profile = record.get("aggregate_profile")
                if profile is not None:
                    if (profile["version"] != 1 or profile["calls"] != record["aggregate_calls"] or
                            set(profile["host_ns"]) != set(AGGREGATE_STAGES) or
                            not 0 <= profile["successful_calls"] <= profile["calls"]):
                        raise ValueError("invalid aggregate profile version/coverage")
                    measured = 0
                    for stage in AGGREGATE_STAGES:
                        elapsed = profile["host_ns"][stage]
                        if not isinstance(elapsed, int) or elapsed < 0:
                            raise ValueError("invalid aggregate stage time")
                        samples[f"{phase}.host.aggregate_detail.{stage}"].append(elapsed)
                        measured += elapsed
                    samples[f"{phase}.host.aggregate_detail.wrapper"].append(
                        difference(record["aggregate_host_ns"], measured))
                    for field in ("calls", "successful_calls", "new_pages", "copied_bytes", "channel_elements"):
                        counters[f"{phase}.aggregate_profile.{field}"] += profile[field]
                samples[f"{phase}.device.total"].append(difference(stamps["cq_end"], stamps["sq_seen"]))
                if "input_slot_wait_ns" in record:
                    samples[f"{phase}.device.input_slot_wait_ns"].append(record["input_slot_wait_ns"])
    if not rows or len(used) != len(calls):
        raise ValueError("no target writes, or target writes missing packet labels")
    report = dict(version=1, pwrite_calls=len(rows),
                single_command_calls=sum(row["commands"] == 1 for row in rows),
                multi_command_calls=sum(row["commands"] != 1 for row in rows),
                device_correlated=bool(device_records), rows=rows, counters=dict(counters),
                metrics={name: summarize(values) for name, values in sorted(samples.items())},
                notes=[
                    "Async aggregation is reported separately; input-slot backpressure is inside device dispatch, not a transfer or software CPU cost. Same-SQ head-of-line delays before observation remain unclassified.",
                    "Foreground device complements do not subtract background aggregation; do not label their full duration pure software overhead.",
                    "Guest times use trace_clock=mono; text timestamps have microsecond resolution.",
                    "Only single-command calls have additive exclusive Guest partitions.",
                    "request.* sums for split calls overlap; they are not a syscall partition.",
                    "guest_outside_pending_nvme is syscall time outside the union of SQ..CQ spans, not all software work.",
                    "sq_cq_roundtrip includes SQ copy/doorbell, device work, delivery and CQ observation.",
                    "bio_wakeup_dio includes scheduler waiting; wall intervals are not CPU time.",
                    "Device virtual and Host aggregate time are separate domains; never subtract them from Guest timestamps.",
                    "aggregate_detail stages partition Host aggregate time with wrapper remainder, including profiling overhead; layout stages include lookup/allocation/zeroing/copy/enqueue, not copy alone.",
                    "device.check_span includes check_ns and retry_gap; aggregate_ns is inside check_ns.",
                    "Retry intervals are time until rechecking after each reason, including poller/queue delay, not pure NAND wait.",
                    "Tracing adds work; compare with an untraced run. This does not calibrate clocks.",
                ])
    if account_device:
        report["device_accounting"] = device_accounting(pairs, phase_trace, attention_records)
    phases = {(packet["session"], packet["packet_id"]): packet["phase"] for packet in packets}
    asynchronous = {(record["session"], record["packet_id"]) for record in device_records
                    if record.get("input") and record.get("async_aggregate") and
                    (record["session"], record["packet_id"]) in phases}
    if asynchronous:
        seen = set()
        background = defaultdict(lambda: defaultdict(list))
        for record in device_records:
            if record.get("source") != "background_aggregate":
                continue
            key = (record["session"], record["packet_id"])
            if key not in asynchronous:
                continue
            if key in seen or record["status"]:
                raise ValueError("duplicate or failed background aggregation")
            seen.add(key)
            metrics = background[phases[key]]
            metrics["dispatch_virtual_ns"].append(difference(record["started_ns"], record["accepted_ns"]))
            metrics["span_virtual_ns"].append(difference(record["finished_ns"], record["started_ns"]))
            metrics["operator_virtual_ns"].append(record["aggregate_ns"])
            metrics["operator_host_ns"].append(record["aggregate_host_ns"])
        if seen != asynchronous:
            raise ValueError("missing background aggregation completion; capture through final DRAIN")
        report["background_aggregation"] = {
            phase: {name: summarize(values) for name, values in metrics.items()}
            for phase, metrics in background.items()}
        report["notes"].append("Background span includes resource waiting; sums can overlap foreground IO and are not additive to syscall or phase time.")
    return report


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def analyze_directory(args):
    directory = Path(args.directory)
    metadata = json.loads((directory / "capture.json").read_text())
    if metadata["trace_clock"] != "mono" or metadata["lost_events"] or metadata["reader_errors"]:
        raise ValueError("invalid/incomplete trace capture")
    if metadata["clocksource_before"] != metadata["clocksource_after"]:
        raise ValueError("Guest clocksource changed during capture")
    if metadata["command_exit"] != 0:
        raise ValueError("traced command failed")
    guest_lines = (directory / "guest.trace").read_text().splitlines()
    calls = parse_guest(guest_lines)
    packets = json_lines(directory / "packets.jsonl")
    devices = json_lines(args.device) if args.device else []
    if args.device and not devices:
        raise ValueError("empty device trace")
    if args.account_device and (not devices or metadata["clocksource_before"] != "hpet"):
        raise ValueError("--account-device requires --device and validated Guest HPET capture")
    phase_path = Path(args.phase_trace) if args.phase_trace else directory / "phase-intervals.json"
    phase_trace = json.loads(phase_path.read_text()) if phase_path.exists() else None
    if args.phase_trace and phase_trace is None:
        raise ValueError("requested phase trace is missing")
    if args.phase_trace and not args.account_device:
        raise ValueError("--phase-trace requires --account-device")
    attention = json_lines(args.attention_device) if args.attention_device else []
    if args.attention_device and (not attention or phase_trace is None or not args.account_device):
        raise ValueError("Attention accounting requires nonempty timeline, phase intervals and --account-device")
    report = analyze(calls, packets, devices, args.account_device, phase_trace, attention)
    if args.decode_stack:
        if not args.account_device or phase_trace is None:
            raise ValueError("--decode-stack requires --account-device and phase intervals")
        from decode_stack_accounting import analyze_decode_stack
        report["decode_stack"] = analyze_decode_stack(phase_trace, parse_guest(guest_lines, all_io=True), devices,
                                                      report["device_accounting"]["clock_alignment"])
    report["capture"] = metadata
    output = Path(args.output) if args.output else directory / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for name, metric in report["metrics"].items():
        print(f"[pwrite-stack] {name}: total={metric['total_seconds']:.6f}s "
              f"count={metric['count']} avg={metric['average_us']:.3f}us "
              f"p99={metric['p99_us']:.3f}us max={metric['maximum_us']:.3f}us")
    print(f"[pwrite-stack] calls={report['pwrite_calls']} single={report['single_command_calls']} "
          f"split={report['multi_command_calls']} device_correlated={report['device_correlated']}")
    print(f"[pwrite-stack] counters={json.dumps(report['counters'], sort_keys=True)}")
    if "background_aggregation" in report:
        print(f"[background-aggregation] {json.dumps(report['background_aggregation'], sort_keys=True)}")
    if args.account_device:
        accounting = report["device_accounting"]
        print(f"[pwrite-accounting] alignment={json.dumps(accounting['clock_alignment'])}")
        for phase, boundaries in accounting["syscall_by_phase"].items():
            for boundary, values in boundaries.items():
                print(f"[pwrite-accounting] {phase}.{boundary}: syscall={values['target_ns'] / 1e9:.6f}s "
                      f"outside=[{values['outside_min_ns'] / 1e9:.6f}, {values['outside_max_ns'] / 1e9:.6f}]s "
                      f"fraction=[{values['outside_fraction_min']:.2%}, {values['outside_fraction_max']:.2%}]")
        pipeline = accounting.get("pipeline")
        if pipeline:
            for phase, values in pipeline["phases"].items():
                print(f"[phase-accounting] {phase}: window={values['window_ns'] / 1e9:.6f}s")
                for category, item in values["categories"].items():
                    print(f"[phase-accounting] {phase}.{category}: outside_device="
                          f"[{item['outside_min_ns'] / 1e9:.6f}, {item['outside_max_ns'] / 1e9:.6f}]s "
                          f"phase_fraction=[{item['phase_fraction_min']:.2%}, {item['phase_fraction_max']:.2%}]")
            print(f"[phase-accounting] coverage={pipeline['device_coverage']}; "
                  "NOT a complete software-stack fraction or CPU/critical-path attribution")
            request_wait = pipeline["decode_request_wait"]
            if request_wait["status"] == "complete":
                totals = request_wait["totals"]
                print(f"[decode-wait] requests={request_wait['requests']} wait={totals['wait_ns'] / 1e9:.6f}s "
                      f"own-device-uncovered=[{totals['outside_min_ns'] / 1e9:.6f}, {totals['outside_max_ns'] / 1e9:.6f}]s")
                for name, item in totals["categories"].items():
                    print(f"[decode-wait] {name}=[{item['min_ns'] / 1e9:.6f}, {item['max_ns'] / 1e9:.6f}]s")
            else:
                print(f"[decode-wait] unavailable: {request_wait['reason']}")
        else:
            print("[phase-accounting] unavailable: no phase intervals; syscall accounting only")
    print(f"[pwrite-stack] report={output}; SQ/CQ roundtrip is NOT pure software CPU time")
    if "decode_stack" in report:
        stack = report["decode_stack"]
        print(f"[decode-stack] requests={stack['requests']} window={stack['decode_window_ns'] / 1e9:.6f}s")
        print(f"[decode-stack] alignment=SQ..nvme_complete_rq; "
              f"cq_seen_alignment_anomalies={len(stack['cq_seen_alignment_anomalies'])}; "
              "all matched commands retained")
        for name in ("software_union", "software_not_overlapped_by_tracked_device"):
            item = stack[name]
            print(f"[decode-stack] {name}=[{item['min_ns'] / 1e9:.6f}, {item['max_ns'] / 1e9:.6f}]s "
                  f"fraction=[{item['fraction_min']:.2%}, {item['fraction_max']:.2%}]")
        for kind, entry in stack["by_kind"].items():
            item = entry["software_union"]
            print(f"[decode-stack] {kind}: calls={entry['calls']} "
                  f"software=[{item['min_ns'] / 1e9:.6f}, {item['max_ns'] / 1e9:.6f}]s "
                  f"decode_fraction=[{item['fraction_min']:.2%}, {item['fraction_max']:.2%}]")
        if "broad_stack" in stack:
            broad = stack["broad_stack"]
            print(f"[vllm-stack] Decode excluding diagnostic reads={broad['decode_without_trace_ns'] / 1e9:.6f}s")
            for name, item in broad.items():
                if isinstance(item, dict) and "fraction_min" in item:
                    print(f"[vllm-stack] {name}=[{item['min_ns'] / 1e9:.6f}, {item['max_ns'] / 1e9:.6f}]s "
                          f"fraction=[{item['fraction_min']:.2%}, {item['fraction_max']:.2%}]")


def record(args):
    if os.geteuid() != 0:
        raise ValueError("record must run as root inside the Guest")
    if os.environ.get("CEMU_IO_BACKEND", "ioctl") != "ioctl":
        raise ValueError("this recorder measures synchronous pwrite, not io_uring")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("supply -- followed by the inference command")
    clock_path = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
    clocksource = clock_path.read_text().strip()
    if clocksource != "hpet" and not args.allow_other_clock:
        raise ValueError("select validated HPET before recording, or use --allow-other-clock for diagnostics")
    root = Path(args.tracefs)
    missing = [event for event in EVENTS if not (root / "events" / event / "enable").exists()]
    if missing:
        raise ValueError(f"missing tracepoints {missing}; build/boot the updated linux-cemu kernel")
    if "name=" not in (root / "events/fdmfs/fdmfs_rw_begin/format").read_text():
        raise ValueError("old FDMFS trace format; boot the updated linux-cemu kernel")
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    instance = root / "instances" / ("cemu-pwrite-" + uuid.uuid4().hex[:12])
    instance.mkdir()
    stopped = threading.Event()
    errors = []
    reader = None
    descriptor = None
    exit_code = -1
    metadata = dict(version=1, command=command, kernel=os.uname().release, trace_clock="mono",
                    clocksource_before=clocksource, lost_events=0, reader_errors=errors)
    try:
        (instance / "tracing_on").write_text("0")
        (instance / "trace_clock").write_text("mono")
        (instance / "buffer_size_kb").write_text(str(args.buffer_kb))
        for event in EVENTS:
            (instance / "events" / event / "enable").write_text("1")
        descriptor = os.open(instance / "trace_pipe", os.O_RDONLY | os.O_NONBLOCK)

        def consume():
            try:
                with (directory / "guest.trace").open("wb") as output, selectors.DefaultSelector() as selector:
                    selector.register(descriptor, selectors.EVENT_READ)
                    while True:
                        selector.select(timeout=.1)
                        try:
                            data = os.read(descriptor, 1024 * 1024)
                        except BlockingIOError:
                            if stopped.is_set():
                                break
                            continue
                        if data:
                            output.write(data)
                        elif stopped.is_set():
                            break
            except BaseException as error:
                errors.append(str(error))

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        environment = os.environ.copy()
        environment["CEMU_PWRITE_PACKET_TRACE"] = str(directory / "packets.jsonl")
        environment["CEMU_PROFILE_INTERVALS"] = str(directory / "phase-intervals.json")
        (instance / "tracing_on").write_text("1")
        print(f"[pwrite-stack] capture={directory}", flush=True)
        exit_code = subprocess.call(command, env=environment)
    finally:
        (instance / "tracing_on").write_text("0")
        stopped.set()
        if reader is not None:
            reader.join(timeout=10)
            if reader.is_alive():
                errors.append("trace reader did not drain")
        if descriptor is not None:
            os.close(descriptor)
        stats = {}
        for path in (instance / "per_cpu").glob("cpu*/stats"):
            stats[path.parent.name] = path.read_text()
            for name, value in re.findall(r"^([^:\n]+):\s*(\d+)", stats[path.parent.name], re.MULTILINE):
                if name.strip() in ("overrun", "commit overrun", "dropped events"):
                    metadata["lost_events"] += int(value)
        metadata.update(command_exit=exit_code, cpu_stats=stats,
                        clocksource_after=clock_path.read_text().strip())
        (directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (instance / "events/enable").write_text("0")
        instance.rmdir()
    if errors or metadata["lost_events"]:
        raise ValueError("trace incomplete; inspect capture.json")
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Trace Guest FDM pwrite software/protocol boundaries without BCC")
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture = subparsers.add_parser("record")
    capture.add_argument("--output", required=True)
    capture.add_argument("--tracefs", default="/sys/kernel/tracing")
    capture.add_argument("--buffer-kb", type=int, default=8192)
    capture.add_argument("--allow-other-clock", action="store_true")
    capture.add_argument("command", nargs=argparse.REMAINDER)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("directory")
    analysis.add_argument("--device")
    analysis.add_argument("--output")
    analysis.add_argument("--account-device", action="store_true",
                          help="deduplicate device intervals and bound outside-device elapsed time (HPET only)")
    analysis.add_argument("--phase-trace", help="defaults to directory/phase-intervals.json when present")
    analysis.add_argument("--attention-device", help="optional SparF timeline from this same QEMU run")
    analysis.add_argument("--decode-stack", action="store_true",
                          help="require full pread/pwrite/EXEC command coverage and compute Decode software-path union")
    args = parser.parse_args()
    try:
        if args.action == "record":
            raise SystemExit(record(args))
        analyze_directory(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"[pwrite-stack] FAIL/INCOMPLETE: {error}\n")


if __name__ == "__main__":
    main()
