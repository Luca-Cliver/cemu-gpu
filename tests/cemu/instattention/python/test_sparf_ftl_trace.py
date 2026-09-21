import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/analyze_sparf_ftl_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_sparf_ftl_trace", MODULE_PATH)
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


class SparfFtlTraceTest(unittest.TestCase):
    def summarize(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text("\n".join(records) + "\n")
            return TRACE.summarize(path)

    def record(self, **overrides):
        fields = dict(op="write", controller=0, nsid=1, cookie=123, kind=0,
                      stream=0, group=0, page_in_group=0, ch=0, lun=0, pl=0, blk=0, pg=0)
        fields.update(overrides)
        return " ".join(f"{key}={value}" for key, value in fields.items())

    def test_kv_and_untagged_read_waits_are_separate(self):
        summary = self.summarize([
            self.record(op="read", kind=0, wait_ns=10),
            self.record(op="read", kind=1, wait_ns=20),
            self.record(op="read", kind=2, wait_ns=30),
            self.record(op="read", kind=255, cookie=0, wait_ns=1000),
        ])
        self.assertEqual(summary["reads"], 4)
        self.assertEqual(summary["kv_reads"], 3)
        self.assertEqual(summary["untagged_reads"], 1)
        self.assertEqual(summary["read_wait_ns"]["total"], 1060)
        self.assertEqual(summary["kv_read_wait_ns"], dict(total=60, max=30, avg=20))
        self.assertEqual(summary["untagged_read_wait_ns"]["total"], 1000)
        self.assertEqual(summary["records_without_identity"], 0)

    def test_devices_namespaces_sessions_and_planes_are_not_conflated(self):
        summary = self.summarize([
            self.record(),
            self.record(controller=1),
            self.record(nsid=2),
            self.record(cookie=456, pg=1),
            self.record(pl=1),
        ])
        self.assertEqual(len(summary["groups"]), 4)
        self.assertEqual(summary["groups"]["0/1/123/0/0/0"]["pages"], 2)
        self.assertEqual(summary["physical_pages_reused"], 0)

    def test_physical_reuse_is_scoped_to_device_not_session(self):
        summary = self.summarize([self.record(), self.record(cookie=456)])
        self.assertEqual(len(summary["groups"]), 2)
        self.assertEqual(summary["physical_pages_reused"], 1)

    def test_legacy_trace_explicitly_has_unknown_identity(self):
        legacy = "kind=0 stream=0 group=0 page_in_group=0 ch=0 lun=0 blk=0 pg=0"
        summary = self.summarize([legacy, "op=read " + legacy + " wait_ns=7"])
        self.assertEqual(summary["writes"], 1)
        self.assertEqual(summary["records_without_identity"], 2)
        self.assertIn("unknown/unknown/unknown/0/0/0", summary["groups"])
        self.assertEqual(summary["kv_read_wait_ns"]["total"], 7)

    def test_empty_trace(self):
        summary = self.summarize([])
        self.assertEqual(summary["records"], 0)
        self.assertEqual(summary["kv_read_wait_ns"], dict(total=0, max=0, avg=0))
        self.assertEqual(summary["groups"], {})


if __name__ == "__main__":
    unittest.main()
