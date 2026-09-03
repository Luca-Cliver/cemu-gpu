#!/usr/bin/env python3

import csv
import sys
import tempfile
import time
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from experiments import PipelineTrace


class PipelineTraceTest(unittest.TestCase):
    def setUp(self):
        self.trace = PipelineTrace()
        self.base = time.perf_counter_ns()

    def record(self, offset_ns, message, thread="test-thread"):
        self.trace.record(
            message,
            timestamp_ns=self.base + offset_ns,
            thread_name=thread,
        )

    def test_prefill_overlap(self):
        self.record(0, "[prefill] layer=0 MLP-start")
        self.record(10, "[prefill-kv-writer] store-start layer=0, tokens=8")
        self.record(20, "[prefill] layer=1 Attention-start")
        self.record(60, "[prefill] layer=0 MLP-complete hidden=(1, 8, 16)")
        self.record(80, "[prefill-kv-writer] store-complete layer=0, tokens=8")
        self.record(90, "[prefill] layer=1 Attention-complete hidden=(1, 8, 16)")

        overlaps = self.trace.overlaps()
        measured = {
            (item.first_operation, item.second_operation): item.overlap_ns
            for item in overlaps
        }
        self.assertEqual(measured[("store", "mlp")], 50)
        self.assertEqual(measured[("store", "attention")], 60)

    def test_decode_overlap(self):
        self.record(
            100,
            "[cemu-attention-slot-scheduler] compute-start request=0, "
            "slot=0, layer=0, tokens=9",
        )
        self.record(
            110,
            "[decode] QKV-start position=8, layer=1",
            thread="MainThread",
        )
        self.record(
            120,
            "[kv-backend] store-start request=0, layer=0, token=8",
            thread="cemu-attention-store_0",
        )
        self.record(
            130,
            "[cemu-attention-slot-scheduler] prefetch-start request=1, "
            "slot=1, layer=1, tokens=8",
            thread="cemu-attention-load_0",
        )
        self.record(
            160,
            "[decode] QKV-complete position=8, layer=1",
            thread="MainThread",
        )
        self.record(
            170,
            "[cemu-attention-slot-scheduler] prefetch-complete request=1, "
            "slot=1, layer=1, tokens=8",
            thread="cemu-attention-load_0",
        )
        self.record(
            180,
            "[kv-backend] store-complete request=0, layer=0, token=8",
            thread="cemu-attention-store_0",
        )
        self.record(
            200,
            "[cemu-attention-slot-scheduler] compute-complete request=0, "
            "slot=0, layer=0, tokens=9",
        )

        measured = {
            (item.first_operation, item.second_operation): item.overlap_ns
            for item in self.trace.overlaps()
        }
        self.assertEqual(measured[("compute", "store")], 60)
        self.assertEqual(measured[("compute", "load")], 40)
        self.assertEqual(measured[("qkv", "store")], 40)

    def test_write_csv_files(self):
        self.record(0, "[prefill] layer=0 MLP-start")
        self.record(10, "[prefill] layer=0 MLP-complete hidden=(1, 1, 1)")
        with tempfile.TemporaryDirectory() as directory:
            event_path, overlap_path = self.trace.write(
                Path(directory) / "pipeline.csv"
            )
            self.assertTrue(event_path.is_file())
            self.assertTrue(overlap_path.is_file())
            with event_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["thread"], "test-thread")


if __name__ == "__main__":
    unittest.main()
