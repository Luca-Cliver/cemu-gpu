import copy
import importlib.util
import random
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("pwrite_stack_trace",
    Path(__file__).resolve().parents[1] / "experiments/pwrite_stack_trace.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def line(microseconds, event, body="", tid=42):
    return f" python3-{tid} [001] .... 1.{microseconds:06d}: {event}{': ' if body else ''}{body}\n"


def command(stage, cid=7, size=512, offset=4096):
    return f"stage={stage} ctrl=0 qid=1 cid={cid} nsid=2 opcode=5 offset={offset} bytes={size}"


def fixture():
    lines = [
        line(0, "sys_pwrite64(fd: 3, buf: 1234, count: 200, pos: 0)"),
        line(5, "fdmfs_rw_begin", "iocb abc offset=0 bytes=512 name=kv_buffer_data"),
        line(10, "iomap_dio_rw_begin", "dev 1:0"),
        line(20, "nvme_cemu_io", command("prepared")),
        line(25, "nvme_cemu_io", command("sq_copy")),
        line(200, "nvme_cemu_io", command("cq_seen"), tid=0),
        line(205, "nvme_complete_rq", "nvme0: qid=1, cmdid=7, status=0x0", tid=0),
        line(240, "iomap_dio_complete", "error 0 ret 512"),
        line(245, "fdmfs_rw_end", "iocb abc"),
        line(250, "sys_pwrite64 -> 0x200"),
    ]
    packet = dict(version=1, tid=42, session=900, packet_id=1, phase="prefill",
                  bytes=512, start_ns=999999000, end_ns=1000251000, success=True)
    stamps = dict(sq_seen=10, memory_enter=11, locked=12, dma_begin=13, dma_end=15,
                  memory_done=16, model_ready=20, first_check=22, last_check=40,
                  cq_begin=41, cq_end=42)
    device = dict(version=1, controller=0, qid=1, cid=7, nsid=2, offset=4096,
                  bytes=512, input=True, session=900, packet_id=1, status=0, virtual=stamps,
                  check_ns=8, aggregate_ns=3, aggregate_host_ns=3000,
                  space_retry_ns=10, busy_retry_ns=0, drain_retry_ns=0, model_retry_ns=0,
                  checks=2, aggregate_calls=2, space_retries=1, busy_retries=0, cq_full=0)
    return lines, packet, device


class PwriteStackTraceTest(unittest.TestCase):
    def test_nvme_transfer_tags_preserve_existing_time_accounting(self):
        lines, packet, device = fixture()
        tagged = [entry.rstrip("\n") + " transfer_id=91 packet_bytes=512\n"
                  if "nvme_cemu_io" in entry else entry for entry in lines]
        self.assertEqual(TRACE.analyze(TRACE.parse_guest(lines), [packet], [device]),
                         TRACE.analyze(TRACE.parse_guest(tagged), [packet], [device]))

    def test_vllm_phase_windows_do_not_require_flexgen_future(self):
        from test_sparf_autonomous_timeline import sample

        names = [("vllm.prefill.step_wall", 0, 10), ("vllm.decode.step_wall", 10, 600),
                 ("terminal_drain.wall", 600, 700), ("decode.execute_wait_wall", 10, 540)]
        trace = dict(version=1, clock="clock_gettime(CLOCK_MONOTONIC)", clocksource="hpet", stack=True,
                     intervals=[dict(name=name, start_ns=start, end_ns=end) for name, start, end in names])
        report = TRACE.phase_accounting(trace, [], [], dict(offset_lower_ns=0, offset_upper_ns=0), sample(), 1)
        self.assertEqual(report["phases"]["decode"]["window_ns"], 590)
        self.assertEqual(report["decode_request_wait"]["status"], "not_applicable")
        self.assertNotIn("mixed", report["phases"])

    def test_async_aggregation_separate_from_acknowledged_io(self):
        lines, packet, device = fixture()
        device.update(async_aggregate=True, aggregate_calls=0, aggregate_ns=0, aggregate_host_ns=0)
        background = dict(source="background_aggregate", input=False, session=900, packet_id=1,
                          status=0, accepted_ns=40, started_ns=50, finished_ns=100,
                          aggregate_ns=30, aggregate_host_ns=3000)
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device, background])
        self.assertEqual(report["counters"]["prefill.device.aggregate_calls"], 0)
        self.assertEqual(report["background_aggregation"]["prefill"]["span_virtual_ns"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "missing background"):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])
        with self.assertRaisesRegex(ValueError, "duplicate or failed"):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [device, background, background])

    def request_fixture(self):
        entries = [dict(name="framework.decode_iterations_wall", start_ns=90, end_ns=230)]
        contexts = [dict(id="first", layer=0, microbatch=0, tokens=1025, program=2),
                    dict(id="second", layer=0, microbatch=1, tokens=1025, program=3)]

        def add(context, name, start, end):
            entries.append(dict(name=name, start_ns=start, end_ns=end, tid=42, request=context))

        first, second = contexts
        add(first, "framework.attention_result_wait_wall", 100, 200)
        add(first, "decode.attention.queue_wait_wall", 100, 110)
        add(first, "decode.attention.worker_wall", 110, 180)
        add(first, "decode.store.lock_wait_wall", 110, 120)
        add(first, "decode.execute_wait_wall", 120, 160)
        add(first, "decode.output_read_wall", 160, 170)
        add(second, "framework.attention_result_wait_wall", 200, 220)
        add(second, "decode.attention.queue_wait_wall", 70, 75)
        add(second, "decode.attention.worker_wall", 75, 205)
        add(second, "decode.execute_wait_wall", 90, 195)
        completed = [dict(program=2, layer=0, tokens=1025, request=21,
                          started=dict(virtual_ns=130), ready=dict(virtual_ns=150)),
                     dict(program=3, layer=0, tokens=1025, request=22,
                          started=dict(virtual_ns=100), ready=dict(virtual_ns=190))]
        pairs = []
        for context in contexts:
            call = dict(enter=80, exit=88)
            row = dict(phase="decode", request=context)
            record = dict(checks=0, virtual=dict(memory_enter=81, memory_done=85, model_ready=86))
            pairs.append((call, row, [(None, record)]))
        return dict(request_context_version=1, intervals=entries), pairs, completed

    def test_request_wait_does_not_use_other_slots_device_coverage(self):
        trace, pairs, completed = self.request_fixture()
        report = TRACE.decode_wait_accounting(trace, pairs, completed,
                                             dict(offset_lower_ns=0, offset_upper_ns=0))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["totals"]["wait_ns"], 120)
        self.assertEqual(report["totals"]["outside_min_ns"], 100)
        first = report["rows"][0]
        self.assertEqual(first["outside_min_ns"], 80)
        self.assertEqual(first["categories"]["worker_queue"]["min_ns"], 10)
        self.assertEqual(first["categories"]["lock_wait"]["min_ns"], 10)
        self.assertEqual(first["categories"]["exec_transport_or_device_queue"]["min_ns"], 20)
        self.assertEqual(first["categories"]["return_to_waiter"]["min_ns"], 20)
        for row in report["rows"]:
            self.assertEqual(sum(item["min_ns"] for item in row["categories"].values()), row["outside_min_ns"])
            self.assertEqual(row["outside_min_ns"], row["outside_max_ns"])

    def test_request_wait_bounds_nested_spans_and_completed_future(self):
        trace, pairs, completed = self.request_fixture()
        trace["intervals"].append(dict(name="decode.store_append_wall", start_ns=100, end_ns=180,
                                      request=trace["intervals"][1]["request"]))
        report = TRACE.decode_wait_accounting(trace, pairs, completed,
                                             dict(offset_lower_ns=-2, offset_upper_ns=3))
        self.assertEqual(report["rows"][0]["outside_min_ns"], 75)
        self.assertEqual(report["rows"][0]["outside_max_ns"], 85)
        self.assertEqual(report["rows"][1]["outside_min_ns"], 20)
        for item in report["totals"]["categories"].values():
            self.assertLessEqual(item["min_ns"], item["max_ns"])
        for entry in trace["intervals"]:
            if entry["name"] == "framework.attention_result_wait_wall" and entry["request"]["id"] == "second":
                entry.update(start_ns=210, end_ns=211)
        report = TRACE.decode_wait_accounting(trace, pairs, completed,
                                             dict(offset_lower_ns=0, offset_upper_ns=0))
        self.assertEqual(report["rows"][1]["categories"]["return_to_waiter"]["min_ns"], 1)

    def test_request_match_handles_slot_reuse_without_global_fifo(self):
        trace, pairs, completed = self.request_fixture()
        for entry in trace["intervals"]:
            if entry.get("request", {}).get("id") == "second":
                entry["request"]["program"] = 2
                if entry["name"] == "decode.execute_wait_wall":
                    entry.update(start_ns=170, end_ns=195)
        completed[1].update(program=2, started=dict(virtual_ns=175), ready=dict(virtual_ns=185))
        report = TRACE.decode_wait_accounting(trace, pairs, list(reversed(completed)),
                                             dict(offset_lower_ns=0, offset_upper_ns=0))
        self.assertEqual([row["device_request"] for row in report["rows"]], [21, 22])

    def test_request_wait_rejects_missing_or_ambiguous_identity(self):
        for failure in ("packet", "execute", "ambiguous", "wait", "overlap", "coordinates"):
            trace, pairs, completed = self.request_fixture()
            if failure == "packet":
                pairs[0][1]["request"] = None
            elif failure == "execute":
                next(entry for entry in trace["intervals"] if entry["name"] == "decode.execute_wait_wall")["request"] = None
            elif failure == "ambiguous":
                completed.append(copy.deepcopy(completed[0]))
            elif failure == "wait":
                trace["intervals"] = [entry for entry in trace["intervals"]
                                      if entry["name"] != "framework.attention_result_wait_wall"]
            elif failure == "overlap":
                trace["intervals"][7]["start_ns"] = 150
            else:
                pairs[0][1]["request"] = dict(pairs[0][1]["request"], microbatch=7)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                TRACE.decode_wait_accounting(trace, pairs, completed,
                                             dict(offset_lower_ns=0, offset_upper_ns=0))

    def test_cli_offline_accounting_and_hpet_guard(self):
        lines, packet, device = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(trace_clock="mono", lost_events=0, reader_errors=[], command_exit=0,
                            clocksource_before="hpet", clocksource_after="hpet")
            (root / "capture.json").write_text(json.dumps(metadata))
            (root / "guest.trace").write_text("".join(lines))
            (root / "packets.jsonl").write_text(json.dumps(packet) + "\n")
            (root / "device.jsonl").write_text(json.dumps(device) + "\n")
            command_line = [sys.executable, str(SPEC.origin), "analyze", directory,
                            "--device", str(root / "device.jsonl"), "--account-device"]
            result = subprocess.run(command_line, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("phase intervals", result.stdout)
            report = json.loads((root / "summary.json").read_text())
            self.assertIn("device_accounting", report)
            metadata.update(clocksource_before="kvm-clock", clocksource_after="kvm-clock")
            (root / "capture.json").write_text(json.dumps(metadata))
            result = subprocess.run(command_line, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HPET", result.stderr)

    def test_attention_device_coverage_checks_count_pid_and_time(self):
        from test_sparf_autonomous_timeline import sample

        def span(name, start, end):
            return dict(name=name, start_ns=start, end_ns=end, tid=42)

        trace = dict(version=1, clock="clock_gettime(CLOCK_MONOTONIC)", clocksource="hpet", stack=True,
            intervals=[span("framework.prefill_iteration_wall", 0, 10),
                       span("framework.decode_iterations_wall", 10, 600),
                       span("pipeline.generate_wall", 0, 600),
                       span("pipeline.final_flush_wall", 600, 700),
                       span("decode.execute_wait_wall", 10, 540),
                       span("framework.attention_result_wait_wall", 10, 550)])
        alignment = dict(offset_lower_ns=0, offset_upper_ns=0)
        report = TRACE.phase_accounting(trace, [], [], alignment, sample(), 1)
        self.assertEqual(report["attention_requests"], 1)
        waiting = report["phases"]["decode"]["categories"]["main_thread_blocking_wait"]
        self.assertEqual(waiting["outside_min_ns"], 40)
        with self.assertRaisesRegex(ValueError, "PID/count"):
            TRACE.phase_accounting(trace, [], [], alignment, sample(), 2)
        with self.assertRaisesRegex(ValueError, "generation interval"):
            TRACE.phase_accounting(trace, [], [], dict(offset_lower_ns=1000, offset_upper_ns=1000), sample(), 1)
        with self.assertRaisesRegex(ValueError, "missing completion"):
            TRACE.phase_accounting(trace, [], [], alignment, sample()[:1], 1)
        context = dict(id="decode:4:0:16", layer=4, microbatch=0, tokens=17, program=3)
        trace["request_context_version"] = 1
        for entry in trace["intervals"]:
            if entry["name"] in ("decode.execute_wait_wall", "framework.attention_result_wait_wall"):
                entry["request"] = context
        trace["intervals"].extend([
            dict(span("decode.attention.worker_wall", 10, 545), request=context),
            dict(span("decode.attention.queue_wait_wall", 5, 10), request=context)])
        pairs = [(dict(enter=11, exit=15), dict(phase="decode", request=context),
                  [(None, dict(checks=0, virtual=dict(memory_enter=12, memory_done=13, model_ready=14)))])]
        report = TRACE.phase_accounting(trace, pairs, [], alignment, sample(), 1)
        self.assertEqual(report["decode_request_wait"]["totals"]["outside_min_ns"], 38)

    def test_interval_operations_against_discrete_reference(self):
        randomizer = random.Random(123)
        for _ in range(100):
            left = [tuple(sorted(randomizer.sample(range(30), 2))) for _ in range(5)]
            right = [tuple(sorted(randomizer.sample(range(30), 2))) for _ in range(5)]
            left_set = {tick for start, end in left for tick in range(start, end)}
            right_set = {tick for start, end in right for tick in range(start, end)}
            self.assertEqual(TRACE.interval_size(left), len(left_set))
            self.assertEqual(TRACE.interval_size(TRACE.interval_intersection(left, right)), len(left_set & right_set))
            self.assertEqual(TRACE.interval_size(TRACE.interval_subtract(left, right)), len(left_set - right_set))
            alignment = dict(offset_lower_ns=-3, offset_upper_ns=4)
            bounds = TRACE.outside_device(left, right, alignment)
            for offset in range(-3, 5):
                shifted = {tick + offset for tick in right_set}
                actual = len(left_set - shifted)
                self.assertLessEqual(bounds["outside_min_ns"], actual)
                self.assertGreaterEqual(bounds["outside_max_ns"], actual)

    def test_device_accounting_deduplicates_overlapping_commands(self):
        _, packet, device = fixture()
        first = dict(key=(0, 1, 7), sq_copy=1_000_000, cq_seen=1_200_000)
        second = dict(first, key=(0, 1, 8))
        call = dict(enter=900_000, exit=1_300_000)
        row = dict(phase="prefill")
        report = TRACE.device_accounting([(call, row, [(first, device), (second, device)])])
        accounting = row["device_accounting"]
        self.assertEqual(accounting["command_service_sum_ns"], 54)
        self.assertEqual(accounting["service_union_ns"], 27)
        self.assertEqual(accounting["service_and_wait"]["outside_min_ns"], 400000 - 27)
        self.assertEqual(accounting["service_and_wait"]["outside_max_ns"], 400000 - 27)
        self.assertEqual(report["syscall_by_phase"]["prefill"]["request_envelope"]["outside_min_ns"], 400000 - 31)

    def test_accounting_is_opt_in_and_weighted_by_syscall_duration(self):
        lines, packet, device = fixture()
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device], account_device=True)
        self.assertIn("device_accounting", report)
        metric = report["device_accounting"]["syscall_by_phase"]["prefill"]["service_and_wait"]
        self.assertEqual(metric["target_ns"], 250000)
        self.assertAlmostEqual(metric["outside_fraction_min"], (250000 - 27) / 250000)
        self.assertNotIn("device_accounting", TRACE.analyze(TRACE.parse_guest(lines), [packet], [device]))
        longer = TRACE.parse_guest(lines)[0]
        longer["exit"] += 250000
        row = dict(phase="prefill")
        pairs = [(longer, row, [(longer["commands"][0], device)]),
                 (TRACE.parse_guest(lines)[0], dict(phase="prefill"), [(longer["commands"][0], device)])]
        metric = TRACE.device_accounting(pairs)["syscall_by_phase"]["prefill"]["service_and_wait"]
        self.assertEqual(metric["target_ns"], 750000)
        self.assertAlmostEqual(metric["outside_fraction_min"], (750000 - 54) / 750000)

    def test_clock_offset_mismatch_and_multiple_sessions_are_rejected(self):
        lines, packet, device = fixture()
        calls = TRACE.parse_guest(lines)
        later = copy.deepcopy(calls[0])
        later["commands"][0]["sq_copy"] += 1_000_000_000
        later["commands"][0]["cq_seen"] += 1_000_000_000
        pairs = [(calls[0], dict(phase="prefill"), [(calls[0]["commands"][0], device)]),
                 (later, dict(phase="decode"), [(later["commands"][0], device)])]
        with self.assertRaisesRegex(ValueError, "clock offset"):
            TRACE.device_accounting(pairs)
        pairs[1] = (calls[0], dict(phase="decode"), [(calls[0]["commands"][0], dict(device, session=901))])
        with self.assertRaisesRegex(ValueError, "clock offset"):
            TRACE.device_accounting(pairs)

    def test_phase_windows_clip_spilled_prefill_and_keep_unknowns(self):
        def span(name, start, end):
            return dict(name=name, start_ns=start, end_ns=end, tid=42)

        trace = dict(version=1, clock="clock_gettime(CLOCK_MONOTONIC)", clocksource="hpet", stack=True,
            intervals=[span("framework.prefill_iteration_wall", 0, 100),
                       span("framework.decode_iterations_wall", 100, 200),
                       span("pipeline.final_flush_wall", 200, 230),
                       span("prefill.store_append_wall", 20, 120),
                       span("prefill.fdm_write.call_wall", 40, 110),
                       span("prefill.writer_backpressure_wall", 30, 90),
                       span("framework.attention_result_wait_wall", 100, 180)])
        pairs = [(dict(enter=50, exit=105), dict(phase="prefill"), [])]
        alignment = dict(offset_lower_ns=0, offset_upper_ns=0)
        report = TRACE.phase_accounting(trace, pairs, [(60, 103)], alignment, [], None)
        self.assertIsNone(report["software_stack_fraction"])
        self.assertIn("MISSING", report["device_coverage"])
        prefill, decode = report["phases"]["prefill"], report["phases"]["decode"]
        self.assertEqual(prefill["categories"]["append_syscalls"]["target_ns"], 50)
        self.assertEqual(decode["categories"]["append_syscalls"]["target_ns"], 5)
        self.assertEqual(prefill["categories"]["client_preparation"]["target_ns"], 30)
        self.assertEqual(decode["categories"]["client_preparation"]["target_ns"], 15)
        self.assertEqual(prefill["categories"]["main_thread_blocking_wait"]["outside_min_ns"], 30)
        self.assertEqual(decode["categories"]["main_thread_blocking_wait"]["outside_min_ns"], 77)
        for bad in (dict(trace, dropped=1), dict(trace, clocksource="kvm-clock"), dict(trace, stack=False)):
            with self.assertRaises(ValueError):
                TRACE.phase_accounting(bad, pairs, [], alignment, [], None)
        with self.assertRaisesRegex(ValueError, "missing phase"):
            TRACE.phase_accounting(dict(trace, intervals=[]), pairs, [], alignment, [], None)

    def test_aggregate_profile_partition_and_retry_coverage(self):
        lines, packet, device = fixture()
        device["aggregate_profile"] = dict(version=1, calls=2, successful_calls=1,
            new_pages=18, copied_bytes=768, channel_elements=128,
            host_ns={stage: 100 for stage in TRACE.AGGREGATE_STAGES})
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])
        metrics = report["metrics"]
        total = sum(metrics["prefill.host.aggregate_detail." + stage]["total_seconds"]
                    for stage in (*TRACE.AGGREGATE_STAGES, "wrapper"))
        self.assertAlmostEqual(total, metrics["prefill.host.aggregate_ns"]["total_seconds"])
        self.assertEqual(report["counters"]["prefill.aggregate_profile.new_pages"], 18)
        device["aggregate_profile"]["calls"] = 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])
        device["aggregate_profile"]["calls"] = 2
        device["aggregate_profile"]["host_ns"]["token_k"] = 4000
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])

    def test_single_call_partition_and_cross_thread_completion(self):
        lines, packet, device = fixture()
        calls = TRACE.parse_guest(lines)
        report = TRACE.analyze(calls, [packet], [device])
        self.assertEqual(report["pwrite_calls"], 1)
        self.assertEqual(report["single_command_calls"], 1)
        self.assertEqual(sum(report["rows"][0]["exclusive_guest_ns"].values()), 250000)
        self.assertAlmostEqual(report["metrics"]["prefill.single.sq_cq_roundtrip"]["average_us"], 175)
        self.assertAlmostEqual(report["metrics"]["prefill.device.retry_gap"]["total_seconds"], 10e-9)
        self.assertAlmostEqual(report["metrics"]["prefill.host.aggregate_ns"]["total_seconds"], 3000e-9)
        self.assertEqual(report["counters"]["prefill.device.space_retries"], 1)
        self.assertAlmostEqual(report["metrics"]["prefill.device.space_retry_ns"]["total_seconds"], 10e-9)

    def test_retry_reason_accounting_is_checked(self):
        lines, packet, device = fixture()
        with self.assertRaisesRegex(ValueError, "retry reason"):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [dict(device, space_retry_ns=9)])

    def test_aggregation_yield_is_separate_from_resource_wait(self):
        lines, packet, device = fixture()
        device.update(space_retry_ns=4, aggregate_yield_ns=6)
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])
        self.assertAlmostEqual(report["metrics"]["prefill.device.aggregate_yield_ns"]["total_seconds"], 6e-9)

    def test_unfinished_packet_command_has_no_aggregation(self):
        lines, packet, device = fixture()
        device.update(checks=0, check_ns=0, aggregate_ns=0, aggregate_host_ns=0,
                      space_retry_ns=0, aggregate_calls=0, space_retries=0)
        device["virtual"].update(first_check=0, last_check=0)
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device])
        self.assertAlmostEqual(report["metrics"]["prefill.device.ready_to_cq"]["total_seconds"], 21e-9)

    def test_split_commands_do_not_claim_exclusive_partition(self):
        lines, packet, device = fixture()
        lines[1] = line(5, "fdmfs_rw_begin", "iocb abc offset=0 bytes=1024 name=kv_buffer_data")
        lines.insert(5, line(30, "nvme_cemu_io", command("prepared", cid=8, offset=4608)))
        lines.insert(6, line(35, "nvme_cemu_io", command("sq_copy", cid=8, offset=4608)))
        lines.insert(9, line(210, "nvme_cemu_io", command("cq_seen", cid=8, offset=4608), tid=0))
        lines.insert(10, line(215, "nvme_complete_rq", "nvme0: qid=1, cmdid=8, status=0x0", tid=0))
        lines[-1] = line(250, "sys_pwrite64 -> 0x400")
        packet["bytes"] = 1024
        second = copy.deepcopy(device)
        second.update(cid=8, offset=4608)
        report = TRACE.analyze(TRACE.parse_guest(lines), [packet], [device, second])
        self.assertEqual(report["multi_command_calls"], 1)
        self.assertNotIn("exclusive_guest_ns", report["rows"][0])
        self.assertEqual(report["metrics"]["prefill.request.sq_to_cq"]["count"], 2)
        self.assertEqual(report["rows"][0]["guest_roundtrip_union_ns"], 185000)
        self.assertEqual(report["rows"][0]["guest_outside_pending_nvme_ns"], 65000)

    def test_reused_cid_and_multiple_phases(self):
        lines, packet, device = fixture()
        later = [entry.replace("1.", "2.", 1) for entry in lines]
        second = dict(packet, packet_id=2, phase="decode",
                      start_ns=1999999000, end_ns=2000251000)
        second_device = dict(device, packet_id=2)
        report = TRACE.analyze(TRACE.parse_guest(lines + later), [packet, second],
                               [device, second_device])
        self.assertEqual(report["pwrite_calls"], 2)
        self.assertIn("decode.syscall_wall", report["metrics"])

    def test_reject_missing_or_ambiguous_packet(self):
        lines, packet, _ = fixture()
        for packets in ([], [dict(packet, tid=99)], [packet, packet]):
            with self.assertRaises(ValueError):
                TRACE.analyze(TRACE.parse_guest(lines), packets)

    def test_reject_missing_cq_failed_and_short_writes(self):
        lines, packet, _ = fixture()
        for entries in ([entry for entry in lines if "cq_seen" not in entry],
                        [entry.replace("status=0x0", "status=0x2") for entry in lines],
                        lines[:-1] + [line(250, "sys_pwrite64 -> 0x100")]):
            with self.assertRaises((ValueError, KeyError)):
                TRACE.analyze(TRACE.parse_guest(entries), [packet])

    def test_device_mismatch_is_not_silently_subtracted(self):
        lines, packet, device = fixture()
        with self.assertRaises(ValueError):
            TRACE.analyze(TRACE.parse_guest(lines), [packet], [dict(device, cid=9)])

    def test_unfinished_and_lost_events(self):
        lines, _, _ = fixture()
        with self.assertRaises(ValueError):
            TRACE.parse_guest(lines[:-1])
        with self.assertRaises(ValueError):
            TRACE.parse_guest(["CPU:1 [LOST 4 EVENTS]\n"])

    def test_old_kernel_and_other_files_do_not_match(self):
        lines, packet, _ = fixture()
        lines = [entry.replace("name=kv_buffer_data", "name=query") for entry in lines]
        self.assertEqual(TRACE.parse_guest(lines), [])
        with self.assertRaises(ValueError):
            TRACE.analyze([], [packet])

    def test_named_syscall_format(self):
        lines, packet, _ = fixture()
        lines[0] = line(0, "sys_enter_pwrite64", "fd: 3")
        lines[-1] = line(250, "sys_exit_pwrite64", "ret: 0x200")
        self.assertEqual(TRACE.analyze(TRACE.parse_guest(lines), [packet])["pwrite_calls"], 1)


if __name__ == "__main__":
    unittest.main()
