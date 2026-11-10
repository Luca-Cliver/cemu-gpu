"""Admission ordering and failure handling without a GPU or vLLM installation."""

from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from vllm_adapter.run import drain_session, generate_requests
from vllm_adapter.session import Config, Session


class EarlySchedulingEngine:
    def __init__(self):
        self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(call_utility=self.utility))
        self.paused = False
        self.pending = []
        self.first_batch = None
        self.events = []

    def utility(self, method, *args):
        self.events.append((method, *args))
        if method == "pause_scheduler":
            self.paused = True
        elif method == "resume_scheduler":
            self.paused = False
            self.step()
        else:
            raise AssertionError(method)

    def step(self):
        if not self.paused and self.first_batch is None:
            self.first_batch = tuple(self.pending)

    def enqueue(self, prompts, sampling_params):
        for prompt in prompts:
            self.pending.append(prompt)
            self.events.append(("add", prompt))
            if len(self.pending) == 2:
                self.step()
        return list(self.pending)

    def wait_for_completion(self):
        self.events.append(("wait",))
        self.step()
        return self.first_batch

    def generate(self, prompts, sampling_params):
        self.enqueue(prompts, sampling_params)
        return self.wait_for_completion()


class AdmissionTest(unittest.TestCase):
    def test_unbarriered_generate_can_start_b2_before_all_b8_requests_arrive(self):
        engine = EarlySchedulingEngine()
        self.assertEqual(generate_requests(engine, list(range(8)), object()), (0, 1))

    def test_fixed_batch_waits_for_all_adds_before_resume(self):
        engine = EarlySchedulingEngine()
        self.assertEqual(generate_requests(engine, list(range(8)), object(), fixed_batch_size=8),
                         tuple(range(8)))
        self.assertEqual(engine.events, [("pause_scheduler", "keep", False)] +
                         [("add", index) for index in range(8)] + [("resume_scheduler",), ("wait",)])

    def test_partial_or_failed_enqueue_never_runs_a_partial_cohort(self):
        for response in (["a"], ["a", "a"], ValueError("enqueue failed")):
            with self.subTest(response=response):
                engine = EarlySchedulingEngine()
                engine.enqueue = Mock()
                if isinstance(response, Exception):
                    engine.enqueue.side_effect = response
                else:
                    engine.enqueue.return_value = response
                with self.assertRaises((RuntimeError, ValueError)):
                    generate_requests(engine, ["a", "b"], object(), fixed_batch_size=2)
                self.assertTrue(engine.paused)
                self.assertIsNone(engine.first_batch)

    def test_unsupported_client_or_wrong_count_never_enqueues(self):
        engine = EarlySchedulingEngine()
        with self.assertRaisesRegex(ValueError, "prompt count"):
            generate_requests(engine, ["a"], object(), fixed_batch_size=8)
        self.assertEqual(engine.events, [])
        engine.llm_engine.engine_core = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "pinned vLLM"):
            generate_requests(engine, ["a"], object(), fixed_batch_size=1)
        self.assertEqual(engine.events, [])

    def test_secondary_drain_failure_preserves_generation_exception(self):
        engine = Mock()
        engine.collective_rpc.side_effect = RuntimeError("engine dead")
        original = ValueError("unmeasured Softmax shape")
        diagnostic = StringIO()
        with redirect_stderr(diagnostic), self.assertRaises(ValueError) as caught:
            try:
                raise original
            finally:
                drain_session(engine, failed=True)
        self.assertIs(caught.exception, original)
        self.assertIn("engine dead", diagnostic.getvalue())

    def test_drain_failure_after_success_is_not_suppressed(self):
        engine = Mock()
        engine.collective_rpc.side_effect = RuntimeError("drain failed")
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            drain_session(engine, failed=False)
        engine.collective_rpc.side_effect = None
        engine.collective_rpc.return_value = [{"closed": True}]
        self.assertEqual(drain_session(engine, failed=False), [{"closed": True}])


class FixedBatchContractTest(unittest.TestCase):
    def test_guard_rejects_smaller_or_mixed_batches_before_device_access(self):
        store, device = Mock(), Mock()
        session = Session(Config("unused", "unused", "unused", request_capacity=8, fixed_batch_size=8),
                          1, 40, 40, 128, 1.0, store=store, device=device)
        for starts, prompts, counts in (([0] * 2, [1024] * 2, [1024] * 2),
                                       ([1024] + [0] * 7, [1024] * 8, [1] + [1024] * 7)):
            with self.subTest(starts=starts), self.assertRaisesRegex(ValueError, "fixed-batch experiment"):
                session.plan(list(range(len(starts))), starts, prompts, counts)
        self.assertEqual(session.step, 0)
        self.assertEqual(session._requests, {})
        self.assertEqual(store.mock_calls, [])
        self.assertEqual(device.mock_calls, [])
        rows = session.plan(list(range(8)), [0] * 8, [1024] * 8, [1024] * 8)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row.prefill for row in rows))
        session._counts = {(slot, 0): 1024 for slot in range(8)}
        rows = session.plan(list(reversed(range(8))), [1024] * 8, [1024] * 8, [1] * 8)
        self.assertEqual([row.slot for row in rows], list(reversed(range(8))))
        self.assertFalse(any(row.prefill for row in rows))

    def test_fixed_batch_is_opt_in_and_validated(self):
        self.assertIsNone(Config("unused", "unused", "unused").fixed_batch_size)
        for size in (0, -1, True, 1.5, 9):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "fixed_batch_size"):
                Config("unused", "unused", "unused", request_capacity=8, fixed_batch_size=size)


if __name__ == "__main__":
    unittest.main()
