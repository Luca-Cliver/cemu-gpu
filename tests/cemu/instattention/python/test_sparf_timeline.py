import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_sparf_timeline import chrome_trace, overlap_ns, read_records, summarize
from analyze_sparf_timeline import virtual_trace
from sparf_model_diagnostics import validate_diagnostics


def command(program, stage, start, end, events=()):
    return dict(schema=1, clock="host_monotonic_ns", pid=1, session=100,
                command=program * 10 + stage, controller=0, program=program,
                workflow=1, stage=stage, layer=0, tokens=129, worker=int(stage >= 3),
                workers=2, compute_units=1, success=True, worker_start_ns=start,
                start_ns=start, end_ns=end, worker_end_ns=end, modeled_ns=10,
                intervals=[dict(name=name, start_ns=a, end_ns=b) for name, a, b in events])


class TimelineTest(unittest.TestCase):
    def with_model(self, row, received, start, finish):
        row["virtual"] = dict(clock="qemu_virtual_ns", functional_start_ns=received,
                              functional_end_ns=received)
        row["reads"] = []
        completion = {k: row[k] for k in ("pid", "session", "controller", "command")}
        completion.update(schema=1, event="completion", clock="qemu_virtual_ns", cu=0,
                          received_ns=received, start_ns=start, finish_ns=finish,
                          scheduled_runtime_ns=finish-start)
        return completion

    def test_virtual_submission_can_overlap_despite_serial_host_execution(self):
        # Host execution is disjoint, but both requests were in flight in the
        # virtual system. CU execution itself remains serialized in this fixture.
        a, b = command(1, 0, 0, 10), command(2, 0, 20, 30)
        completions = [self.with_model(a, 100, 100, 200), self.with_model(b, 150, 200, 300)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [completions[1], a, b, completions[0]]))
            rows = read_records(path)
        report, = summarize(rows)
        self.assertEqual(report["load_compute_overlap_ns"], 0)
        model = report["model"]
        self.assertEqual(model["scheduler"]["next_received_before_previous_finish"], 1)
        self.assertEqual(model["scheduler"]["queue_sum_ns"], 50)
        self.assertEqual(model["scheduler"]["submission_gap_median_ns"], -50)
        self.assertEqual(model["virtual_functional_advance_sum_ns"], 0)
        self.assertTrue(all(e["cat"] == "qemu_virtual" for e in virtual_trace(rows)["traceEvents"]))

    def test_read_amplification_and_lun_imbalance(self):
        row = command(1, 0, 0, 10)
        self.with_model(row, 100, 100, 200)
        row["reads"] = [dict(name="token_k", logical_bytes=512, submitted_bytes=8192,
            ranges=2, model_start_ns=100, model_end_ns=300, pages=2, unmapped_pages=0,
            service_ns=100, queue_ns=250, max_queue_ns=150, critical_queue_ns=150,
            critical_service_ns=50, critical_initial_wait_ns=100, channels=2,
            luns_per_channel=1, page_bytes=4096, page_read_ns=50, lun_pages=[2, 0])]
        validate_diagnostics(row)
        report, = summarize([row])
        read = report["model"]["reads"]["token_k"]
        self.assertEqual(read["page_amplification"], 16)
        self.assertEqual(read["lun_max_to_mean_median"], 2)
        self.assertEqual(read["model_ns"], 200)  # Not service sum + queue sum.
        self.assertEqual(report["model"]["scheduler"]["missing_completions"], 1)
        row["reads"][0]["lun_pages"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "LUN"):
            validate_diagnostics(row)

    def test_virtual_stage_gaps_are_separate_from_job_runtime(self):
        rows = [command(1, stage, stage * 10, stage * 10 + 5) for stage in range(1, 5)]
        for row, start in zip(rows, (100, 130, 160, 190)):
            row["scheduler"] = self.with_model(row, start, start + 2, start + 12)
        model = summarize(rows)[0]["model"]
        self.assertEqual(model["scheduler"]["stage_boundary_gap_sum_ns"], 54)
        self.assertEqual(model["scheduler"]["queue_sum_ns"], 8)
        self.assertEqual(model["scheduler"]["charged_runtime_sum_ns"], 40)
        self.assertEqual(model["scheduler"]["microbatch_pairs"], 0)

    def test_completion_integrity_and_old_log_compatibility(self):
        row = command(1, 0, 0, 10)
        self.assertFalse(summarize([row])[0]["model"]["available"])
        completion = self.with_model(row, 100, 110, 200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            for rows, error in (([completion], "without"),
                                ([row, completion, completion], "duplicate")):
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaisesRegex(ValueError, error):
                    read_records(path)
            completion["start_ns"] = 99
            path.write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, "timestamps"):
                read_records(path)

    def test_counts_only_cross_workflow_union_in_same_layer_token(self):
        span = lambda a, b, identity, layer=0: (a, b, identity, layer, 129)
        self.assertEqual(overlap_ns([span(0, 10, "a"), span(5, 15, "c")],
                                    [span(7, 12, "b")]), 5)
        self.assertEqual(overlap_ns([span(0, 10, "a")], [span(2, 8, "a")]), 0)
        self.assertEqual(overlap_ns([span(0, 10, "a")], [span(2, 8, "b", 1)]), 0)
        self.assertEqual(overlap_ns([span(0, 10, "a")], [span(10, 20, "b")]), 0)

    def test_known_overlap_and_trace_roundtrip(self):
        records = [command(1, 1, 0, 10), command(1, 2, 10, 20),
                   command(1, 3, 20, 40, [("v_outstanding", 21, 39), ("qk_functional", 25, 32)]),
                   command(1, 4, 40, 45), command(2, 1, 15, 25),
                   command(2, 2, 25, 35, [("token_k_read_wait", 25, 35)]),
                   command(2, 3, 45, 55), command(2, 4, 55, 60)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in reversed(records)))
            loaded = read_records(path)
        report, = summarize(loaded)
        self.assertEqual(report["complete_workflows"], 2)
        self.assertEqual(report["load_compute_overlap_ns"], 15)
        self.assertEqual(report["read_wait_qk_overlap_ns"], 7)
        self.assertEqual(report["incomplete_or_failed_workflows"], 0)
        exported = chrome_trace(loaded)
        self.assertEqual(len(exported["traceEvents"]), 11)

    def test_one_workflow_v_qk_nesting_does_not_prove_cross_slot_overlap(self):
        records = [command(1, 1, 0, 10), command(1, 2, 10, 20),
                   command(1, 3, 20, 40, [("v_outstanding", 20, 40), ("qk_functional", 25, 35)]),
                   command(1, 4, 40, 50)]
        report, = summarize(records)
        self.assertEqual(report["load_compute_overlap_ns"], 0)
        self.assertEqual(report["read_wait_qk_overlap_ns"], 0)

    def test_sessions_and_controllers_are_not_combined(self):
        rows = [command(1, 1, 0, 20), command(2, 3, 0, 20)]
        rows[1]["controller"] = 1
        reports = summarize(rows)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(r["load_compute_overlap_ns"] == 0 for r in reports))

    def test_rejects_duplicate_out_of_order_or_malformed_records(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([command(1, 1, 0, 10)] * 2)
        with self.assertRaisesRegex(ValueError, "out of order"):
            summarize([command(1, 1, 10, 20), command(1, 2, 0, 10)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"schema":1')
            with self.assertRaises(ValueError):
                read_records(path)

    def test_failure_is_not_reported_as_a_complete_workflow(self):
        row = command(1, 1, 0, 10)
        row["success"] = False
        report, = summarize([row])
        self.assertEqual(report["failed_commands"], 1)
        self.assertEqual(report["complete_workflows"], 0)
        self.assertEqual(report["incomplete_or_failed_workflows"], 1)


if __name__ == "__main__":
    unittest.main()
