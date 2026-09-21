from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_sparf_handoffs import admission_summary, covered_ns, summarize_handoffs


def command(program, stage, received, start, finish, session=1):
    return dict(pid=1, session=session, controller=0, program=program, workflow=1,
        command=program * 10 + stage, stage=stage, layer=0, tokens=1025, success=True,
        scheduler=dict(received_ns=received, start_ns=start, finish_ns=finish))


class HandoffTest(unittest.TestCase):
    def test_coverage_is_clipped_union_not_sum(self):
        self.assertEqual(covered_ns(100, 600, [(90, 150), (130, 200), (300, 700)]), 400)
        self.assertEqual(covered_ns(100, 600, [(700, 800)]), 0)

    def test_peer_prestart_does_not_become_cu_service(self):
        rows = [command(9, 1, 50, 60, 100), command(9, 2, 600, 601, 610),
                command(10, 1, 102, 592, 595)]
        report, = summarize_handoffs(rows)
        gap, = report["longest"]
        self.assertEqual(gap["duration_ns"], 500)
        self.assertEqual(gap["peer_received_to_start_cover_ns"], 490)
        self.assertEqual(gap["observed_cu_job_cover_ns"], 3)
        self.assertEqual(gap["peers"][0]["program"], 10)
        self.assertIsNone(report["admission"])

    def test_sessions_and_token_positions_are_not_mixed(self):
        rows = [command(9, 1, 50, 60, 100), command(9, 2, 600, 601, 610),
                command(10, 1, 102, 592, 595, session=2)]
        unrelated = command(11, 1, 102, 592, 595)
        unrelated["tokens"] = 1026
        rows.append(unrelated)
        report = summarize_handoffs(rows)[0]
        self.assertEqual(report["longest"][0]["peer_received_to_start_cover_ns"], 0)

    def test_dispatch_splits_clocks_and_preserves_missing_records(self):
        row = command(9, 1, 100, 130, 140)
        row["dispatch"] = dict(host_clock="host_monotonic_ns", virtual_clock="qemu_virtual_ns",
            host_entry_ns=1000, host_mirror_start_ns=1020, host_mirror_end_ns=1060,
            host_enqueued_ns=1100, virtual_entry_ns=104, virtual_enqueued_ns=120)
        result = admission_summary([row, command(10, 1, 200, 210, 220)])
        self.assertEqual(result, dict(observed_commands=1, host_mirror_prepare_ns=40,
            host_other_dispatch_ns=60, virtual_received_to_handler_ns=4,
            virtual_handler_to_enqueued_ns=16, virtual_enqueued_to_start_ns=10))
        for name, value in [("host_mirror_end_ns", 1010), ("virtual_enqueued_ns", 150),
                            ("host_clock", "qemu_virtual_ns")]:
            invalid = deepcopy(row)
            invalid["dispatch"][name] = value
            with self.assertRaises(ValueError):
                admission_summary([invalid])

    def test_out_of_order_stage_arrival_is_rejected(self):
        rows = [command(9, 1, 50, 60, 100), command(9, 2, 99, 101, 110)]
        with self.assertRaisesRegex(ValueError, "before previous"):
            summarize_handoffs(rows)


if __name__ == "__main__":
    unittest.main()
