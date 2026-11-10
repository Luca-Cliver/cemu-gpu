"""CPU contract tests for the actual adapter forward, without model weights.

vLLM base classes and CUDA transfers/events are test doubles. Tensor operations
and Session bookkeeping are real; this does not test vLLM imports or CUDA.
The separate test_vllm_cemu suite exercises the real QEMU device workflow.
"""

from contextlib import ExitStack, nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from vllm_adapter.session import Config, Session
from phase_profiler import PhaseProfiler


class BackendContractTest(unittest.TestCase):
    kv_heads = 2
    layer_name = "model.layers.0.self_attn.attn"

    def test_interval_context_tracks_reordered_batch_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = PhaseProfiler(stack=True)
            profiler.interval_path = str(Path(directory) / "phases.json")
            profiler._interval_limit = 1000
            self.r.session.profiler = profiler
            self.test_prefill_then_reordered_decode_without_initialization_context()
            entries = [entry for entry in profiler._intervals if entry["name"] == "decode.batch_wait_wall"]
            self.assertTrue(entries)
            self.assertEqual([len(entry["request"]["rows"]) for entry in entries], [2, 1])
            for entry in entries:
                context = entry["request"]
                self.assertEqual(context["framework"], "vllm")
                self.assertEqual(context["layer"], 0)
                self.assertEqual(len({row["slot"] for row in context["rows"]}), len(context["rows"]))

    def test_detailed_profiling_preserves_outputs_and_submissions(self):
        self.r.session.profiler = PhaseProfiler(stack=True)
        self.test_prefill_then_reordered_decode_without_initialization_context()
        metrics = {m.name: m for m in self.r.session.profiler.snapshot()}
        for name in ("prefill.attention_host_wall", "decode.attention_host_wall",
                     "prefill.append_wall", "decode.mean_and_rows_wall",
                     "decode.h2d.host_wall", "prefill.dense_attention.host_wall"):
            self.assertGreater(metrics[name].total_ns, 0)

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.appends = []
        self.submissions = []

        class Builder:
            def __init__(self, *args):
                pass

        class Event:
            def record(self):
                pass

            def synchronize(self):
                pass

        def execute(query, layer, tokens, mean, *, batch_rows, output):
            self.submissions.append(batch_rows)
            lengths = np.array([r[1] for r in batch_rows], dtype=np.float16)
            self.assertTrue(np.shares_memory(output, self.r.staging.result_array))
            self.assertEqual(output.ctypes.data % 4096, 0)
            np.copyto(output, (query * .25 + lengths[:, None, None]).astype(np.float16))
            return output

        session = Session(Config("unused", "unused", "unused", request_capacity=2),
                          1, 4, self.kv_heads, 128, 128**-.5,
                          store=SimpleNamespace(append_batch=lambda *args: self.appends.append(args)),
                          device=SimpleNamespace(run_decode=execute))
        self.r = SimpleNamespace(session=session, active=False, work=None, staging=None,
                                 layer_ids={}, max_batch_tokens=8, max_requests=2)
        bases = ModuleType("vllm.v1.attention.backend")
        bases.AttentionBackend = object
        bases.AttentionImpl = object
        bases.AttentionMetadataBuilder = Builder
        bases.AttentionType = SimpleNamespace(DECODER="decoder")
        worker = ModuleType("vllm_adapter.worker")
        worker.runtime = lambda: self.r
        config = ModuleType("vllm.config")
        self.get_current_config = Mock(side_effect=AssertionError("no initialization config in forward"))
        config.get_current_vllm_config = self.get_current_config
        name = "vllm_adapter._backend_contract_test"
        path = Path(__file__).parent / "vllm_adapter/backend.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        stack.enter_context(patch.dict(sys.modules, {
            "vllm.v1.attention.backend": bases, "vllm_adapter.worker": worker,
            "vllm.config": config, name: module,
        }))
        spec.loader.exec_module(module)
        self.module = module
        # CPU copies are synchronous. Keep actual torch SDPA and tensor copies.
        empty, empty_like = torch.empty, torch.empty_like

        def without_pin(fn):
            def allocate(*args, **kwargs):
                kwargs.pop("pin_memory", None)
                return fn(*args, **kwargs)
            return allocate

        stack.enter_context(patch.object(torch, "empty", without_pin(empty)))
        stack.enter_context(patch.object(torch, "empty_like", without_pin(empty_like)))
        stack.enter_context(patch.object(torch.cuda, "Event", Event))
        stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda self: True)))
        stack.enter_context(patch.object(module, "sdpa_kernel", lambda *args: nullcontext()))
        self.impl = module.CemuAttentionImpl(4, 128, 128**-.5, num_kv_heads=self.kv_heads)
        self.layer = SimpleNamespace(layer_name=self.layer_name)
        self.builder = module.CemuAttentionMetadataBuilder(None, [self.layer.layer_name], None, None)
        self.generator = torch.Generator().manual_seed(73)

    def inputs(self, tokens):
        q = torch.randn(tokens, 4, 128, generator=self.generator, dtype=torch.float16)
        k = torch.randn(tokens, self.kv_heads, 128, generator=self.generator, dtype=torch.float16)
        v = torch.randn(tokens, self.kv_heads, 128, generator=self.generator, dtype=torch.float16)
        return q, k, v

    def forward(self, work):
        self.r.active = True
        self.r.work = work
        starts = [w.offset for w in work] + [sum(w.count for w in work)]
        metadata = self.builder.build(0, SimpleNamespace(query_start_loc_cpu=torch.tensor(starts)))
        self.r.work = None  # Match the runner's metadata-build lifetime.
        q, k, v = self.inputs(starts[-1])
        output = torch.empty_like(q)
        self.impl.forward(self.layer, q, k, v, None, metadata, output)
        self.get_current_config.assert_not_called()
        return q, k, v, output

    def test_prefill_then_reordered_decode_without_initialization_context(self):
        work = self.r.session.plan(["a", "b"], [0, 0], [2, 3], [2, 3])
        q, k, v, output = self.forward(work)
        for w in work:
            sl = slice(w.offset, w.offset + w.count)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q[sl].transpose(0, 1)[None], k[sl].transpose(0, 1)[None],
                v[sl].transpose(0, 1)[None], is_causal=True, enable_gqa=True)
            torch.testing.assert_close(output[sl], expected[0].transpose(0, 1))
        staging = self.r.staging
        work = self.r.session.plan(["b", "a"], [3, 2], [3, 2], [1, 1])
        q, _, _, output = self.forward(work)
        torch.testing.assert_close(output, q * .25 + torch.tensor([4, 3])[:, None, None].half())
        self.assertEqual(self.submissions, [[(1, 4, 1), (0, 3, 1)]])
        self.assertIs(self.r.staging, staging)
        work = self.r.session.plan(["b"], [4], [3], [1])
        q, _, _, output = self.forward(work)
        torch.testing.assert_close(output, q * .25 + 5)
        self.assertEqual(self.submissions[-1], [(1, 5, 1)])
        self.assertEqual(len(self.submissions), 2)  # One call per layer, independent of B.
        self.assertEqual(len(self.appends), 5)

    def test_profile_has_no_append_and_real_forward_requires_metadata(self):
        q, k, v = self.inputs(1)
        output = torch.ones_like(q)
        self.impl.forward(self.layer, q, k, v, None, None, output)
        self.assertEqual(output.count_nonzero().item(), 0)
        self.assertEqual(self.appends, [])
        self.assertIsNone(self.r.staging)
        self.r.active = True
        with self.assertRaisesRegex(RuntimeError, "request identity"):
            self.impl.forward(self.layer, q, k, v, None, None, output)

    def test_mixed_one_token_prefill_and_decode(self):
        self.forward(self.r.session.plan(["a"], [0], [2], [2]))
        work = self.r.session.plan(["b", "a"], [0, 2], [1, 2], [1, 1])
        q, _, v, output = self.forward(work)
        torch.testing.assert_close(output[0], v[0].repeat_interleave(4 // self.kv_heads, dim=0))
        torch.testing.assert_close(output[1], q[1] * .25 + 3)
        self.assertEqual(self.submissions, [[(0, 3, 1)]])


class OptBackendContractTest(BackendContractTest):
    kv_heads = 4  # OPT MHA, instead of Llama GQA.
    layer_name = "model.decoder.layers.0.self_attn.attn"


if __name__ == "__main__":
    unittest.main()
