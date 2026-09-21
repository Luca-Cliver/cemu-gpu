from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_sparf_autonomous import SCHEMA, analyze, select_records, summarize_reads, summarize_concurrency


def stamp(time):
    return {"host_ns": 1000000 + time * 3, "virtual_ns": time}


def sample():
    def read(name, submitted, deadline, observed, ready):
        return dict(name=name, submitted=stamp(submitted), deadline_ns=deadline,
                    observed=stamp(observed), ready=stamp(ready))

    def compute(name, functional, submitted, start, end, observed):
        return dict(name=name, scheduled=True, truncated=False, effective_ns=end-start,
                    submitted_ns=submitted, completed_ns=end, functional_start=stamp(functional),
                    functional_end=stamp(functional), observed_done=stamp(observed),
                    cu_spans=[dict(cu=0, start_ns=start, end_ns=end)])

    row = dict(schema=SCHEMA, event="begin", pid=1, session=2, controller=0, request=1,
               program=3, layer=4, tokens=17, host_clock="host_monotonic_ns",
               virtual_clock="qemu_virtual_ns", admitted=stamp(10))
    complete = dict(row, event="complete", success=True, started=stamp(20), ready=stamp(520),
        reads=[read("channel_k", 30, 100, 40, 110), read("token_k", 170, 220, 180, 230),
               read("token_v", 240, 400, 360, 410)],
        computes=[compute("approx", 120, 125, 130, 150, 160),
                  compute("qk", 250, 260, 270, 340, 350),
                  compute("pv", 420, 430, 440, 500, 510)])
    return [row, complete]


class AutonomousTimelineTest(unittest.TestCase):
    def test_concurrent_engines_and_cu_occupancy(self):
        first = sample()
        second = deepcopy(first)
        for row in second:
            row["request"] = 2
        for rows, engine in ((first, 0), (second, 1)):
            for row in rows:
                row.update(engines=2, compute_units=2)
            rows[1]["engine"] = engine
            for phase in rows[1]["computes"]:
                phase["cu_spans"][0]["cu"] = engine
        records = [first[0], second[0], first[1], second[1]]
        self.assertEqual(len(analyze(records)), 2)
        summary = summarize_concurrency(records)
        self.assertEqual(summary["max_active_requests"], 2)
        self.assertEqual(summary["cross_request_compute_ns"], 150)
        self.assertEqual(summary["cross_request_io_compute_ns"], 70)
        second[1]["engine"] = 0
        with self.assertRaisesRegex(ValueError, "escaped"):
            analyze(records)

    def test_single_request_is_not_cross_request_overlap(self):
        summary = summarize_concurrency(sample())
        self.assertEqual(summary["max_active_requests"], 1)
        self.assertEqual(summary["cross_request_compute_ns"], 0)
        self.assertEqual(summary["cross_request_io_compute_ns"], 0)

    def diagnostic_sample(self):
        rows = sample()
        for read in rows[1]["reads"]:
            start = read["submitted"]["virtual_ns"]
            duration = read["deadline_ns"] - start
            initial = duration - 10
            read["io"] = dict(requested_bytes=640, logical_bytes=512, tail_bytes=128,
                tail_copies=64, logical_spans=4, aligned_bytes_before_dedup=16384,
                submitted_bytes=12288, ranges=3, model_start_ns=start,
                model_end_ns=read["deadline_ns"], pages=3, unmapped_pages=0, service_ns=15,
                queue_ns=initial * 2 + 5, max_queue_ns=initial + 5,
                critical_queue_ns=initial + 5, critical_service_ns=5,
                critical_initial_wait_ns=initial, channels=2, luns_per_channel=1,
                page_bytes=4096, page_read_ns=5, lun_pages=[2, 1],
                lun_initial_wait_ns=[initial, 0])
        return rows

    def test_diagnostics_separate_tail_dedup_and_critical_path(self):
        reports = analyze(self.diagnostic_sample())
        data = reports[0]["read_io"]["channel_k"]
        self.assertEqual(data["channel_pages"], [2, 1])
        self.assertEqual(data["critical_subsequent_queue_ns"], 5)
        self.assertEqual(data["model_ns"], 70)
        self.assertNotEqual(data["model_ns"], data["service_ns"] + data["queue_ns"])
        summary = summarize_reads(reports)["channel_k"]
        self.assertEqual(summary["submitted_to_logical_ratio"], 24)
        self.assertEqual(summary["tail_fraction"], 0.2)
        self.assertEqual(summary["dedup_saved_bytes"], 4096)
        self.assertEqual(summary["observed_requests"], 1)
        self.assertEqual(summary["missing_requests"], 0)

    def test_missing_diagnostics_are_not_zero_io(self):
        summary = summarize_reads(analyze(sample()))["token_k"]
        self.assertEqual(summary["observed_requests"], 0)
        self.assertEqual(summary["missing_requests"], 1)
        self.assertIsNone(summary["tail_fraction"])

    def test_tail_only_and_unmapped_reads(self):
        rows = self.diagnostic_sample()
        read = rows[1]["reads"][0]
        read["deadline_ns"] = read["submitted"]["virtual_ns"]
        read["io"] = {name: [] if isinstance(value, list) else 0 for name, value in read["io"].items()}
        read["io"].update(requested_bytes=640, tail_bytes=640, tail_copies=320,
                          model_start_ns=30, model_end_ns=30)
        summary = summarize_reads(analyze(rows))["channel_k"]
        self.assertEqual(summary["tail_fraction"], 1)
        self.assertEqual(summary["pages"], 0)
        self.assertEqual(summary["model_ns"], 0)
        rows[1]["reads"][1]["io"]["unmapped_pages"] = 1
        self.assertEqual(summarize_reads(analyze(rows))["token_k"]["unmapped_pages"], 1)

    def test_inconsistent_diagnostics_fail(self):
        for changes in (dict(lun_pages=[1, 1]), dict(lun_initial_wait_ns=[0]),
                        dict(requested_bytes=1), dict(model_end_ns=999),
                        dict(aligned_bytes_before_dedup=0), dict(critical_initial_wait_ns=999)):
            rows = self.diagnostic_sample()
            rows[1]["reads"][0]["io"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analyze(rows)

    def test_session_and_token_filter_keeps_request_pairs(self):
        older = sample()
        latest = deepcopy(older)
        for row in latest:
            row["session"] = 3
            row["tokens"] = 1025
            row["admitted"]["host_ns"] += 1
        selected = select_records(older + latest, latest_session=True, min_tokens=1025, max_tokens=1040)
        self.assertEqual(selected, latest)
        self.assertEqual(len(analyze(selected)), 1)
        for row in latest:
            row["admitted"]["host_ns"] = 1
        self.assertEqual(select_records(older + latest, latest_session=True), latest)
        with self.assertRaises(ValueError):
            analyze(select_records(older + latest, latest_session=True, min_tokens=1041))

    def test_valid_dependencies_and_separate_clocks(self):
        report, = analyze(sample())
        self.assertEqual(report["qk_v_target_overlap_ns"], 70)
        self.assertEqual(report["cu_occupied_ns"], 150)
        self.assertEqual(report["host_elapsed_ns"], 1530)
        self.assertEqual(report["virtual_elapsed_ns"], 510)

    def test_zero_budget_is_not_an_unobserved_compute(self):
        rows = sample()
        rows[1]["computes"][1].update(scheduled=False, effective_ns=0, cu_spans=[], completed_ns=260)
        report, = analyze(rows)
        self.assertEqual(report["zero_budget_phases"], 1)
        self.assertEqual(report["qk_v_target_overlap_ns"], 0)

    def test_qk_can_outlast_v_read(self):
        rows = sample()
        rows[1]["reads"][2]["deadline_ns"] = 280
        self.assertEqual(analyze(rows)[0]["qk_v_target_overlap_ns"], 10)

    def test_incomplete_duplicate_and_empty_traces_fail(self):
        rows = sample()
        for records in ([], rows[:1], rows[1:], rows + rows, [rows[0], rows[1], rows[1]]):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                analyze(records)

    def test_missing_or_truncated_cu_data_fails(self):
        for change in (dict(cu_spans=[]), dict(truncated=True), dict(effective_ns=999)):
            rows = sample()
            rows[1]["computes"][1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(rows)

    def test_early_pv_or_selected_k_fails(self):
        rows = sample()
        rows[1]["computes"][2]["functional_start"] = stamp(300)
        with self.assertRaisesRegex(ValueError, "QK/V join"):
            analyze(rows)
        rows = sample()
        rows[1]["reads"][1]["submitted"] = stamp(140)
        with self.assertRaisesRegex(ValueError, "approximate compute"):
            analyze(rows)

    def test_io_wait_cannot_contain_parent_cu_occupancy(self):
        rows = sample()
        rows[1]["computes"][0]["cu_spans"][0]["start_ns"] = 80
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_failed_requests_and_wrong_schema_fail(self):
        for field, value in (("success", False), ("schema", 1), ("host_clock", "guest")):
            rows = sample()
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze(rows)

    def test_cross_request_cu_double_booking_fails(self):
        rows = sample()
        second = deepcopy(rows)
        for row in second:
            row["request"] = 2
        with self.assertRaisesRegex(ValueError, "double booked"):
            analyze(rows + second)
        for row in second:
            row["controller"] = 1
        self.assertEqual(len(analyze(rows + second)), 2)


if __name__ == "__main__":
    unittest.main()
