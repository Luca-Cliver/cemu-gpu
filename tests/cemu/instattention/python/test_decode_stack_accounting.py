import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from decode_stack_accounting import analyze_decode_stack, CALLS
from pwrite_stack_trace import parse_guest
from test_pwrite_stack_trace import fixture


def stack_fixture(requests=1):
    trace = dict(version=1, request_context_version=1, clock="clock_gettime(CLOCK_MONOTONIC)",
                 clocksource="hpet", stack=True, dropped=0,
                 intervals=[dict(name="framework.decode_iterations_wall", start_ns=0, end_ns=1000)])
    calls, devices = [], []
    for request in range(requests):
        context = dict(id=f"decode:{request}", layer=0, microbatch=request, tokens=1025)
        thread = 42 + request
        for sequence, (name, (kind, syscall, opcode, suffix)) in enumerate(
                (item for item in CALLS.items() if item[1][0] not in ("prefill_append", "drain"))):
            start = 100 + sequence * 100 + request * 10
            identity = (0, request + 1, sequence + 1)
            command = dict(key=identity, nsid=3 if opcode == 1 else 2, opcode=opcode, coverage=2,
                           bytes=512, offset=sequence * 4096, prepared=start + 10, sq_copy=start + 20,
                           cq_seen=start + 60, complete=start + 70, status=0)
            calls.append(dict(tid=thread, syscall=syscall, enter=start, exit=start + 80, commands=[command],
                              returned=0 if opcode == 1 else 512, bytes=512,
                              filename="kv_buffer_data" if kind == "kv_append" else "flexgen_slot0" + (suffix or "")))
            stamps = dict(sq_seen=start + 22, memory_enter=start + 30, locked=start + 31,
                          dma_begin=start + 33, dma_end=start + 35, memory_done=start + 40,
                          model_ready=start + 45, cq_begin=start + 58, cq_end=start + 59,
                          exec_enter=start + 30, exec_admitted=start + 32,
                          exec_started=start + 34, exec_ready=start + 50)
            devices.append(dict(version=2, pid=123, controller=identity[0], qid=identity[1], cid=identity[2],
                                nsid=command["nsid"], opcode=opcode, offset=command["offset"], bytes=512,
                                status=0, virtual=stamps))
            trace["intervals"].append(dict(name=name, start_ns=start - 1, end_ns=start + 81,
                                           tid=thread, request=context))
        trace["intervals"].append(dict(name="framework.attention_result_wait_wall", start_ns=100,
                                       end_ns=700, tid=99, request=context))
    return trace, calls, devices, dict(offset_lower_ns=0, offset_upper_ns=0,
                                     guest_trace_tolerance_ns=0, guest_profile_tolerance_ns=0)


class DecodeStackAccountingTest(unittest.TestCase):
    def vllm_fixture(self, read_trace=False):
        trace, calls, devices, alignment = stack_fixture()
        trace["intervals"][0]["name"] = "vllm.decode.step_wall"
        context = dict(framework="vllm", id="vllm:2:0", step=2, layer=0, read_trace=read_trace,
                       rows=[dict(request_id=str(slot), slot=slot, start=1024, count=1, prefill=False)
                             for slot in range(2)])
        for entry in trace["intervals"][1:]:
            entry["request"] = context
            if entry["name"] == "framework.attention_result_wait_wall":
                entry["name"] = "decode.batch_wait_wall"
        if not read_trace:
            trace["intervals"] = [entry for entry in trace["intervals"]
                                   if entry["name"] != "attention.trace.read.call_wall"]
            calls.pop(4)
            devices.pop(4)
        extra_call = copy.deepcopy(calls[0])
        extra_call.update(enter=20, exit=100)
        command = extra_call["commands"][0]
        command["key"] = (0, 1, 99)
        for field in ("prepared", "sq_copy", "cq_seen", "complete"):
            command[field] -= 80
        extra_device = copy.deepcopy(devices[0])
        extra_device["cid"] = 99
        extra_device["virtual"] = {name: value - 80 for name, value in extra_device["virtual"].items()}
        calls.append(extra_call)
        devices.append(extra_device)
        trace["intervals"].append(dict(name="decode.fdm_write.call_wall", start_ns=19, end_ns=101,
                                       tid=42, request=context))
        return trace, calls, devices, alignment

    def test_vllm_multiple_appends_one_exec_without_diagnostic_read(self):
        report = analyze_decode_stack(*self.vllm_fixture())
        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["by_kind"]["kv_append"]["calls"], 2)
        self.assertEqual(report["by_kind"]["execute"]["calls"], 1)
        self.assertNotIn("trace_read", report["by_kind"])
        self.assertEqual(report["software_union"]["min_ns"], 264)
        broad = report["broad_stack"]
        self.assertEqual(broad["decode_without_trace_ns"], 1000)
        self.assertEqual(broad["broad_software_union"]["min_ns"] +
                         broad["foreground_device_union"]["min_ns"] -
                         broad["software_device_overlap"]["min_ns"] + broad["remaining"]["min_ns"], 1000)

    def test_vllm_diagnostic_exclusion_and_broad_preparation(self):
        trace, calls, devices, alignment = self.vllm_fixture(read_trace=True)
        trace["intervals"] += [dict(name="decode.trace_read_wall", start_ns=499, end_ns=581),
                                dict(name="decode.mean_and_rows_wall", start_ns=0, end_ns=10),
                                dict(name="decode.d2h_wait_wall", start_ns=10, end_ns=19)]
        report = analyze_decode_stack(trace, calls, devices, alignment)
        broad = report["broad_stack"]
        self.assertEqual(broad["diagnostic_excluded_ns"], 82)
        self.assertEqual(broad["preparation"]["min_ns"], 10)
        self.assertEqual(broad["transfer_api_and_wait"]["min_ns"], 9)
        self.assertGreater(broad["broad_software_union"]["min_ns"], broad["syscall_outside_own_device"]["min_ns"])

    def test_payload_child_scopes_do_not_double_count_software_time(self):
        trace, calls, devices, alignment = self.vllm_fixture()
        trace["intervals"].append(dict(name="decode.append_wall", start_ns=0, end_ns=101))
        trace["intervals"].append(dict(name="decode.kv_payload_pack_wall", start_ns=0, end_ns=18))
        expected = analyze_decode_stack(trace, calls, devices, alignment)
        trace["metadata"] = {"decode.kv_payload_layout": {"shape": [1, 40, 128]}}
        for index, name in enumerate(("k_copy_wall", "v_copy_wall", "join_wall")):
            trace["intervals"].append(dict(name="decode.kv_payload_pack." + name,
                                           start_ns=index * 6, end_ns=(index + 1) * 6))
        self.assertEqual(analyze_decode_stack(trace, calls, devices, alignment), expected)

    def test_vllm_missing_append_or_wrong_identity_fails_closed(self):
        trace, calls, devices, alignment = self.vllm_fixture()
        trace["intervals"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            analyze_decode_stack(trace, calls, devices, alignment)
        trace, calls, devices, alignment = self.vllm_fixture()
        trace["intervals"][1]["request"]["rows"][0]["prefill"] = True
        with self.assertRaisesRegex(ValueError, "batch identity"):
            analyze_decode_stack(trace, calls, devices, alignment)

    def test_background_aggregation_is_not_a_second_nvme_completion(self):
        trace, calls, devices, alignment = stack_fixture()
        expected = analyze_decode_stack(trace, calls, devices, alignment)
        background = copy.deepcopy(devices[0])
        background["source"] = "background_aggregate"
        devices.append(background)
        self.assertEqual(analyze_decode_stack(trace, calls, devices, alignment), expected)

    def test_all_six_io_paths_and_exec_queues(self):
        report = analyze_decode_stack(*stack_fixture())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["software_union"]["min_ns"], 6 * 44)
        self.assertEqual(report["software_union"]["fraction_min"], .264)
        self.assertEqual(report["by_kind"]["execute"]["syscall_union_ns"], 80)
        self.assertEqual(report["by_kind"]["execute"]["syscall_outside_fraction_min"], .55)
        self.assertEqual(report["software_union"], report["software_not_overlapped_by_tracked_device"])
        execute = next(row for row in report["rows"] if row["kind"] == "execute")
        self.assertEqual(execute["commands"][0]["device_admission_queue_ns"], 2)
        self.assertEqual(execute["commands"][0]["device_ready_to_cq_ns"], 8)

    def test_two_slots_union_not_sum_and_other_device_overlap(self):
        report = analyze_decode_stack(*stack_fixture(2))
        self.assertEqual(report["software_union"]["min_ns"], 6 * 64)
        self.assertEqual(sum(row["software_min_ns"] for row in report["rows"]), 12 * 44)
        self.assertEqual(report["software_not_overlapped_by_tracked_device"]["min_ns"], 6 * 44)

    def test_clipping_and_clock_uncertainty(self):
        trace, calls, devices, alignment = stack_fixture()
        alignment.update(offset_lower_ns=-2, offset_upper_ns=3)
        report = analyze_decode_stack(trace, calls, devices, alignment)
        self.assertEqual(report["software_union"]["min_ns"], 6 * 39)
        self.assertEqual(report["software_union"]["max_ns"], 6 * 49)
        trace["intervals"][0].update(start_ns=130, end_ns=150)
        report = analyze_decode_stack(trace, calls, devices, alignment)
        self.assertEqual(report["software_union"]["max_ns"], 0)

    def test_reused_cid_matches_by_causal_time_not_file_order(self):
        trace, calls, devices, alignment = stack_fixture(2)
        for call in calls[6:]:
            call["enter"] += 1000
            call["exit"] += 1000
            for command in call["commands"]:
                command["key"] = (0, 1, command["key"][2])
                for key in ("prepared", "sq_copy", "cq_seen", "complete"):
                    command[key] += 1000
        for record in devices[6:]:
            record["qid"] = 1
            record["virtual"] = {key: value + 1000 for key, value in record["virtual"].items()}
        for entry in trace["intervals"]:
            if entry.get("tid") == 43:
                entry["start_ns"] += 1000
                entry["end_ns"] += 1000
        trace["intervals"][0]["end_ns"] = 2000
        report = analyze_decode_stack(trace, calls, list(reversed(devices)), alignment)
        self.assertEqual(report["software_union"]["min_ns"], 12 * 44)

    def test_all_commands_require_one_consistent_clock_offset(self):
        trace, calls, devices, alignment = stack_fixture()
        alignment.update(offset_lower_ns=-10, offset_upper_ns=10)
        devices[0]["virtual"] = {key: value - 9 for key, value in devices[0]["virtual"].items()}
        devices[1]["virtual"] = {key: value + 9 for key, value in devices[1]["virtual"].items()}
        with self.assertRaisesRegex(ValueError, "common Guest/device clock offset"):
            analyze_decode_stack(trace, calls, devices, alignment)

    def test_early_cq_stamp_retained_with_completion_bound(self):
        trace, calls, devices, alignment = stack_fixture()
        original = analyze_decode_stack(trace, calls, devices, alignment)
        call = next(call for call in calls if call["syscall"] == "pread64")
        command = call["commands"][0]
        command["cq_seen"] = command["sq_copy"] + 4
        report = analyze_decode_stack(trace, calls, devices, alignment)
        self.assertEqual(report["software_union"], original["software_union"])
        self.assertEqual(len(report["rows"]), 6)
        self.assertEqual(len(report["cq_seen_alignment_anomalies"]), 1)
        self.assertEqual(report["cq_seen_alignment_anomalies"][0]["kind"], "output_read")
        command["complete"] = command["cq_seen"] + 1
        with self.assertRaisesRegex(ValueError, "device command match"):
            analyze_decode_stack(trace, calls, devices, alignment)

    def test_split_file_syscall_unions_device_intervals(self):
        trace, calls, devices, alignment = stack_fixture()
        call = calls[0]
        original = call["commands"][0]
        original["bytes"] = 256
        devices[0]["bytes"] = 256
        second = dict(original, key=(0, 1, 99), offset=256)
        call["commands"].append(second)
        device = copy.deepcopy(devices[0])
        device.update(cid=99, offset=256)
        devices.append(device)
        report = analyze_decode_stack(trace, calls, devices, alignment)
        self.assertEqual(report["software_union"]["min_ns"], 264)

    def test_incomplete_mixed_and_failed_data_rejected(self):
        for failure in ("drop", "legacy", "kernel", "missing_call", "missing_device", "duplicate_device",
                        "short", "failed", "mixed", "exec_boundary", "filename", "unattributed"):
            trace, calls, devices, alignment = stack_fixture()
            if failure == "drop":
                trace["dropped"] = 1
            elif failure == "legacy":
                devices[0]["version"] = 1
            elif failure == "kernel":
                calls[1]["commands"][0]["coverage"] = 1
            elif failure == "missing_call":
                calls.pop()
            elif failure == "missing_device":
                devices.pop()
            elif failure == "duplicate_device":
                devices.append(copy.deepcopy(devices[0]))
            elif failure == "short":
                calls[0]["returned"] = 256
            elif failure == "failed":
                calls[0]["commands"][0]["status"] = 1
            elif failure == "mixed":
                devices[0]["pid"] = 456
            elif failure == "exec_boundary":
                next(record for record in devices if record["opcode"] == 1)["virtual"]["exec_ready"] = 0
            elif failure == "filename":
                calls[0]["filename"] = "unrelated"
            else:
                calls.append(dict(calls[0], tid=777))
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                analyze_decode_stack(trace, calls, devices, alignment)

    def test_parser_covers_pread_and_ioctl_and_preserves_legacy_filter(self):
        lines, _, _ = fixture()
        read_lines = [line.replace("pwrite64", "pread64").replace("opcode=5", "opcode=2")
                      .replace("name=kv_buffer_data", "name=flexgen_slot0_output") for line in lines]
        self.assertEqual(parse_guest(read_lines), [])
        reads = parse_guest(read_lines, all_io=True)
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0]["syscall"], "pread64")
        self.assertEqual(reads[0]["commands"][0]["opcode"], 2)
        ioctl_lines = [line.replace("pwrite64", "ioctl").replace("opcode=5", "opcode=1")
                       .replace("-> 0x200", "-> 0x0") for line in lines if "fdmfs_" not in line and "iomap_" not in line]
        commands = parse_guest(ioctl_lines, all_io=True)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["syscall"], "ioctl")
        self.assertEqual(commands[0]["returned"], 0)


if __name__ == "__main__":
    unittest.main()
