#!/usr/bin/env python3
import threading
import tempfile
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout
from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler
from flexgen_adapter import SparfAttentionBackend


class Slot:
    staged = True

    def __init__(self, gate=None, loaded=None, fail=False, compute_entered=None,
                 wait_before_load=None):
        self.gate, self.loaded, self.fail = gate, loaded, fail
        self.active = None
        self.closed = True
        self.count = 0
        self.compute_entered = compute_entered
        self.wait_before_load = wait_before_load

    def open(self):
        self.closed = False

    def prepare_decode(self, value):
        if self.active is not None:
            raise AssertionError("slot reused before collect")
        self.active = SimpleNamespace(value=value, completed_stages=0)
        return self.active

    def execute_stage(self, request):
        assert request is self.active
        if request.completed_stages == 1 and self.loaded:
            if self.wait_before_load and not self.wait_before_load.wait(timeout=5):
                raise AssertionError("compute worker did not progress independently")
            self.loaded.set()
        if request.completed_stages == 2 and self.gate:
            if self.compute_entered:
                self.compute_entered.set()
            if not self.gate.wait(timeout=5):
                raise AssertionError("load could not progress while compute was in flight")
        if self.fail:
            self.fail = False
            raise RuntimeError("injected stage failure")
        request.completed_stages += 1

    def collect_decode(self, request):
        assert request.completed_stages == 4
        self.active = None
        self.count += 1
        return request.value

    def abort_decode(self, request):
        self.active = None

    def modeled_runtime_summary(self):
        return self.count * 10, self.count

    def close(self):
        assert self.active is None
        self.closed = True


class SlotSchedulerTest(unittest.TestCase):
    def test_autonomous_parents_can_execute_concurrently(self):
        entered, prepared = threading.Event(), threading.Event()
        rendezvous = threading.Barrier(2, timeout=5)

        class AutonomousSlot(Slot):
            staged = False
            autonomous = True

            def prepare_decode(self, value):
                request = super().prepare_decode(value)
                if value == 1:
                    if not entered.wait(timeout=5):
                        raise AssertionError("first command did not start")
                    prepared.set()
                return request

            def execute_autonomous(self, request):
                if request.value == 0:
                    entered.set()
                    if not prepared.wait(timeout=5):
                        raise AssertionError("next slot preparation was blocked")
                rendezvous.wait()
                request.completed_stages = 4

        slots = (AutonomousSlot(), AutonomousSlot())
        with CemuSparfSlotScheduler(slots) as scheduler:
            first = scheduler.submit_decode(0)
            second = scheduler.submit_decode(1)
            self.assertEqual(first.result(timeout=5), 0)
            self.assertEqual(second.result(timeout=5), 1)
            self.assertEqual(scheduler.modeled_runtime_summary(), (20, 2))
        self.assertTrue(all(slot.closed for slot in slots))

    def test_autonomous_failure_releases_lease(self):
        class AutonomousSlot(Slot):
            staged = False
            autonomous = True

            def execute_autonomous(self, request):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("injected autonomous failure")
                request.completed_stages = 4

        with CemuSparfSlotScheduler((AutonomousSlot(fail=True), AutonomousSlot())) as scheduler:
            with self.assertRaisesRegex(RuntimeError, "injected autonomous"):
                scheduler.submit_decode(0).result(timeout=5)
            for value in range(4):
                self.assertEqual(scheduler.submit_decode(value).result(timeout=5), value)

    def test_mixed_slot_modes_rejected(self):
        autonomous = Slot()
        autonomous.staged = False
        autonomous.autonomous = True
        with self.assertRaisesRegex(ValueError, "same"):
            CemuSparfSlotScheduler((Slot(), autonomous))

    def test_load_and_compute_use_independent_slots(self):
        loaded = threading.Event()
        compute_entered = threading.Event()
        slots = (Slot(gate=loaded, compute_entered=compute_entered),
                 Slot(loaded=loaded, wait_before_load=compute_entered))
        with CemuSparfSlotScheduler(slots) as scheduler:
            first = scheduler.submit_decode("first")
            second = scheduler.submit_decode("second")
            self.assertEqual(first.result(timeout=5), "first")
            self.assertEqual(second.result(timeout=5), "second")
            self.assertEqual(scheduler.modeled_runtime_summary(), (20, 2))
        self.assertTrue(all(slot.closed for slot in slots))

    def test_failure_releases_slot_and_next_request_succeeds(self):
        with CemuSparfSlotScheduler((Slot(fail=True), Slot())) as scheduler:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                scheduler.submit_decode(0).result(timeout=5)
            for index in range(4):
                self.assertEqual(scheduler.submit_decode(index).result(timeout=5), index)

    def test_shared_lifetime_and_backpressure(self):
        scheduler = CemuSparfSlotScheduler((Slot(), Slot()))
        scheduler.open()
        scheduler.open()
        scheduler.close()
        self.assertTrue(scheduler.is_open)
        requests = [scheduler.submit_decode(index) for index in range(10)]
        scheduler.close()
        self.assertFalse(scheduler.is_open)
        self.assertEqual([r.result() for r in requests], list(range(10)))
        with self.assertRaises(RuntimeError):
            scheduler.submit_decode(0)

    def test_reopen_is_rejected_until_teardown_finishes(self):
        scheduler = CemuSparfSlotScheduler((Slot(), Slot())).open()
        entered, resume = threading.Event(), threading.Event()
        release = scheduler.workers.release

        def delayed_release():
            entered.set()
            if not resume.wait(timeout=5):
                raise AssertionError("teardown was not resumed")
            release()

        with patch.object(scheduler.workers, "release", delayed_release):
            with ThreadPoolExecutor(max_workers=1) as closer:
                closing = closer.submit(scheduler.close)
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    with self.assertRaisesRegex(RuntimeError, "closing"):
                        scheduler.open()
                finally:
                    resume.set()
                closing.result(timeout=5)
        with scheduler:
            self.assertEqual(scheduler.submit_decode(12).result(timeout=5), 12)

    def test_backend_releases_its_shared_reference_only_once(self):
        scheduler = CemuSparfSlotScheduler((Slot(), Slot()))
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 4, 1, 1, 4, np.dtype(np.float32)))
        with tempfile.TemporaryDirectory() as directory:
            backends = [SparfAttentionBackend(
                layout, *(Path(directory) / f"mb{index}_{name}" for name in ("k", "c", "v")),
                attention_device=scheduler, replace_existing=True,
            ) for index in range(2)]
            first, second = backends
            try:
                # Closing an unopened backend must not release another user.
                first.open().open()
                second.close()
                self.assertTrue(scheduler.is_open)
                second.open()
                failed = Future()
                failed.set_exception(RuntimeError("injected request failure"))
                first._pending = failed
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    first.close()
                first.close()
                self.assertTrue(second.is_open)
                self.assertFalse(first.store.is_open)
                self.assertEqual(scheduler.submit_decode(7).result(timeout=5), 7)
            finally:
                for backend in backends:
                    backend.close()
            self.assertFalse(scheduler.is_open)


if __name__ == "__main__":
    unittest.main()
