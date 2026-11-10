import json
import math
import os
from pathlib import Path
import sys
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future
from contextlib import contextmanager

import numpy as np
import torch

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from flexgen_adapter.framework import create_model, validate_inputs
from flexgen_adapter.framework_environment import make_environment, make_policy
from flexgen_adapter.framework_import import FRAMEWORK_ROOT, load_framework
from flexgen_adapter.framework_weights import config_from_json, convert_checkpoint
from opt_runtime import OptCheckpointLoader, OptConfig, OptPrefillRunner, OptDecodeRunner, OptGenerationRunner, OptTorchAttentionBackend
from opt_inputs import comparison_case, compare_tokens, decode_output, load_reference, load_tokenizer, prepare_prompt, write_json
from test_opt_runtime import make_state
from phase_profiler import PhaseProfiler


def checkpoint(directory, dimension=8, positions=32, dtype=torch.float32):
    config = OptConfig(name="model", num_hidden_layers=2, max_position_embeddings=positions,
        hidden_size=dimension, num_attention_heads=2, ffn_dim=dimension + 8,
        vocab_size=32, word_embed_proj_dim=dimension, dtype=dtype)
    source = Path(directory) / "model"
    source.mkdir()
    values = dict(model_type="opt", num_hidden_layers=2, max_position_embeddings=positions,
                  hidden_size=dimension, num_attention_heads=2, ffn_dim=dimension + 8,
                  vocab_size=32, word_embed_proj_dim=dimension, torch_dtype=str(dtype).removeprefix("torch."), pad_token_id=1)
    (source / "config.json").write_text(json.dumps(values))
    torch.save(make_state(config), source / "pytorch_model.bin")
    weights = Path(directory) / "converted"
    convert_checkpoint(source, weights)
    return config, config_from_json(source / "config.json"), source, weights


def reference(config, source, prompt, decode, compression=1, top_r=None):
    class SparseReference(OptTorchAttentionBackend):
        def decode(self, layer, query, valid_tokens):
            keys, values = self.layer_cache(layer)
            shape = (valid_tokens, query.shape[0], config.num_attention_heads, config.head_dim)
            result = sparf_attention(query.numpy(), keys.reshape(shape).numpy(), values.reshape(shape).numpy(),
                                     top_r or config.head_dim, math.ceil(valid_tokens / compression))
            return torch.from_numpy(result.output)

    with OptCheckpointLoader(config, source, device="cpu") as loader:
        prefill = OptPrefillRunner(config, loader).run(torch.as_tensor(prompt), collect_kv_cache=True)
        backend = (OptTorchAttentionBackend if compression == 1 else SparseReference)(config, prefill.kv_cache)
        decoded = OptGenerationRunner(OptDecodeRunner(config, loader, backend)).run(
            prefill.next_token_ids, start_position=prompt.shape[1], decode_steps=decode,
            collect_layer_outputs=True)
        tokens = decoded.token_ids
        logits = [prefill.logits[:, -1]] + [step.decode_result.logits[:, -1] for step in decoded.steps]
        return tokens.numpy(), logits


class MemoryBackend:
    def __init__(self, config, batch, barrier=None, compression=1):
        self.layout = SparfKvCacheLayout(KvLayoutConfig(config.num_hidden_layers,
            config.max_seq_len, batch, config.n_head, config.input_dim // config.n_head, config.dtype))
        self.keys = {}
        self.values = {}
        self.appends = []
        self.store = self
        self.sealed = False
        self.barrier = barrier
        self.compression = compression

    def write_prefill(self, layer, keys, values):
        if layer in self.keys:
            raise AssertionError("duplicate Prefill")
        self.keys[layer] = keys.cpu().numpy().copy()
        self.values[layer] = values.cpu().numpy().copy()

    def append_decode(self, layer, token, keys, values):
        assert token == len(self.keys[layer])
        self.appends.append((layer, token))
        self.keys[layer] = np.concatenate((self.keys[layer], keys.cpu().numpy()))
        self.values[layer] = np.concatenate((self.values[layer], values.cpu().numpy()))

    def submit_decode(self, layer, query, valid_tokens):
        assert valid_tokens == len(self.keys[layer])
        assert len(self.keys) == self.layout.config.num_layers
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return sparf_attention(query.cpu().numpy(), self.keys[layer], self.values[layer],
                               self.layout.config.head_dim, max(1, valid_tokens // self.compression)).output

    def wait_decode(self, request):
        return request

    def drain(self):
        self.sealed = True


class FlexGenFrameworkTest(unittest.TestCase):
    def test_mlp_profile_excludes_pending_attention_resolution(self):
        from flexgen_adapter.framework_cache import PendingAttention
        from flexgen.opt_backend import OptTorchDevice

        profiler = PhaseProfiler(stack=True)
        env = make_environment("cpu", profiler=profiler)
        future = Future()
        future.set_result("attention output")
        inputs = SimpleNamespace(shape=(1, 1, 8), dtype=torch.float32, device=env.gpu)
        resolved = SimpleNamespace(data=torch.zeros((1, 1, 8)))
        pending = PendingAttention(SimpleNamespace(profiler=profiler), future, inputs,
                                   Mock(return_value=resolved))

        @contextmanager
        def measured(profiler, name, device):
            self.assertEqual(name, "framework.mlp")
            self.assertIs(pending.resolved, resolved)
            yield

        with patch("flexgen_adapter.framework_environment.profile_tensor_scope", measured), \
                patch.object(OptTorchDevice, "mlp", return_value=resolved) as calculate:
            self.assertIs(env.gpu.mlp(pending, "weights"), resolved)
            calculate.assert_called_once_with(pending, "weights")

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_generate_matches_old_opt_reference(self):
        for microbatches, overlap, stack in ((1, False, False), (1, True, False),
                (2, False, False), (2, True, False), (1, True, True), (2, True, True)):
            with self.subTest(microbatches=microbatches, overlap=overlap, stack=stack), tempfile.TemporaryDirectory() as directory:
                original, config, source, weights = checkpoint(directory)
                prompt = (np.arange(20).reshape(4, 5) % 28 + 4).astype(np.int64)
                expected, logits = reference(original, source, prompt, 3)
                barrier = Barrier(2) if microbatches == 2 else None
                backends = [MemoryBackend(config, 4 // microbatches, barrier) for _ in range(microbatches)]
                profiler = PhaseProfiler(stack=stack)
                env = make_environment("cpu", profiler=profiler)
                observed = []
                env.gpu.output_observer = lambda value: observed.append(value.clone())
                policy = make_policy(4 // microbatches, microbatches, overlap)
                model = create_model(config, env, weights, policy, backends, profiler)
                try:
                    load_framework().timers("generate").reset()
                    tokens = model.generate(validate_inputs(model, prompt, 3), max_new_tokens=4)
                    model.cemu_session.drain()
                    np.testing.assert_array_equal(tokens[:, 5:], expected)
                    for step, expected_logits in enumerate(logits):
                        actual = torch.cat(observed[step * microbatches:(step + 1) * microbatches])
                        torch.testing.assert_close(actual, expected_logits, atol=2e-5, rtol=2e-5)
                    for backend in backends:
                        self.assertEqual(len(backend.appends), 6)
                        self.assertEqual(len(set(backend.appends)), 6)
                        self.assertTrue(backend.sealed)
                        self.assertTrue(all(len(keys) == 8 for keys in backend.keys.values()))
                    if microbatches == 2:
                        self.assertGreaterEqual(model.cemu_session.peak_pending, 2)
                    self.assertEqual(type(model).__module__, "flexgen.flex_opt")
                    self.assertTrue(Path(load_framework().__file__).is_relative_to(FRAMEWORK_ROOT))
                    metrics = {metric.name: metric for metric in profiler.snapshot()}
                    if stack:
                        self.assertEqual(metrics["prefill.writer.queue_wait_wall"].count, 2 * microbatches)
                        self.assertEqual(metrics["decode.attention.queue_wait_wall"].count, 6 * microbatches)
                        self.assertEqual(metrics["decode.qkv.host_wall"].count, 6 * microbatches)
                        self.assertEqual(metrics["framework.mlp.host_wall"].count, 8 * microbatches)
                        self.assertEqual(metrics["framework.lm_head.host_wall"].count, 4 * microbatches)
                    else:
                        self.assertNotIn("prefill.writer.queue_wait_wall", metrics)
                finally:
                    model.cemu_session.close()
                    model.delete_all_weights()

    def test_buffered_phase_trace_preserves_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            original, config, source, weights = checkpoint(directory)
            prompt = np.array([[2, 4, 5, 6, 7]])
            expected, _ = reference(original, source, prompt, 3)
            path = Path(directory) / "phases.json"
            with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": str(path)}):
                profiler = PhaseProfiler(stack=True)
            backend = MemoryBackend(config, 1)
            model = create_model(config, make_environment("cpu", profiler=profiler), weights,
                                 make_policy(1, 1), [backend], profiler)
            try:
                timer = load_framework().timers("generate")
                timer.reset()
                tokens = model.generate(validate_inputs(model, prompt, 3), max_new_tokens=4)
                model.cemu_session.drain()
                np.testing.assert_array_equal(tokens[:, 5:], expected)
                self.assertEqual(len(timer.start_times), 4)
                self.assertEqual(len(timer.stop_times), 4)
                for index, (start, end) in enumerate(zip(timer.start_times, timer.stop_times)):
                    profiler.trace_interval("framework.prefill_iteration_wall" if index == 0 else
                                            "framework.decode_iterations_wall", round(start * 1e9), round(end * 1e9))
                self.assertFalse(path.exists())
                profiler.write_csv(Path(directory) / "profile.csv")
                trace = json.loads(path.read_text())
                self.assertEqual(trace["dropped"], 0)
                names = [entry["name"] for entry in trace["intervals"]]
                self.assertEqual(names.count("framework.prefill_iteration_wall"), 1)
                self.assertEqual(names.count("framework.decode_iterations_wall"), 3)
                self.assertEqual(names.count("framework.prefill_kv_worker_wall"), 2)
                self.assertEqual(names.count("framework.decode_append_wall"), 6)
                waits = [entry for entry in trace["intervals"]
                         if entry["name"] == "framework.attention_result_wait_wall"]
                workers = [entry for entry in trace["intervals"]
                           if entry["name"] == "decode.attention.worker_wall"]
                self.assertEqual(len(waits), 6)
                self.assertEqual({entry["request"]["id"] for entry in waits},
                                 {entry["request"]["id"] for entry in workers})
                self.assertEqual(len({entry["request"]["id"] for entry in waits}), 6)
            finally:
                model.cemu_session.close()
                model.delete_all_weights()

    def test_reject_invalid_inputs_and_second_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, _, weights = checkpoint(directory)
            backend = MemoryBackend(config, 1)
            model = create_model(config, make_environment("cpu"), weights, make_policy(1, 1), [backend])
            try:
                with self.assertRaises(ValueError):
                    model.generate([[1, 4]], max_new_tokens=2)
                model.generate([[2, 4]], max_new_tokens=2)
                with self.assertRaises(RuntimeError):
                    model.generate([[2, 4]], max_new_tokens=2)
            finally:
                model.cemu_session.close()
                model.delete_all_weights()

    def test_weight_loading_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, _, weights = checkpoint(directory)
            with self.assertLogs("flexgen.weights", level="INFO") as captured:
                model = create_model(config, make_environment("cpu"), weights,
                                     make_policy(1, 1), [MemoryBackend(config, 1)])
            try:
                labels = ("InputEmbed", "CemuSelfAttention(layer=1)", "MLP(layer=1)",
                          "CemuSelfAttention(layer=2)", "MLP(layer=2)", "OutputEmbed")
                expected = [f"load weights {stage}: module={index}/{len(labels)} {label}"
                            for index, label in enumerate(labels, 1) for stage in ("begin", "done")]
                self.assertEqual([record.getMessage() for record in captured.records], expected)
            finally:
                model.cemu_session.close()
                model.delete_all_weights()

    def test_prefill_failure_prevents_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, _, weights = checkpoint(directory)
            backend = MemoryBackend(config, 1)
            model = create_model(config, make_environment("cpu"), weights, make_policy(1, 1), [backend])
            try:
                with patch.object(backend, "write_prefill", side_effect=RuntimeError("write failed")), \
                        patch.object(backend, "submit_decode") as execute:
                    with self.assertRaisesRegex(RuntimeError, "write failed"):
                        model.generate([[2, 4]], max_new_tokens=2)
                    execute.assert_not_called()
            finally:
                model.cemu_session.close()
                model.delete_all_weights()
                load_framework().timers("generate").reset()

    def test_checkpoint_rejects_incomplete_weights_and_unsupported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, source, weights = checkpoint(directory)
            (weights / "model-np/decoder.layers.1.fc1.weight").unlink()
            with self.assertRaises(FileNotFoundError):
                create_model(config, make_environment("cpu"), weights, make_policy(1, 1), [MemoryBackend(config, 1)])
            values = json.loads((source / "config.json").read_text())
            values["activation_function"] = "silu"
            (source / "config.json").write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                config_from_json(source / "config.json")

    def test_foreign_flexgen_is_rejected(self):
        with patch.dict("sys.modules", {"flexgen.foreign": SimpleNamespace(__file__="/tmp/other/flexgen/foreign.py")}):
            with self.assertRaises(RuntimeError):
                load_framework()


class FlexGenInputTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_json(self.root / "config.json", dict(vocab_size=32, bos_token_id=7, pad_token_id=1,
                                                   max_position_embeddings=64))

    def test_synthetic_matches_legacy_and_replays(self):
        prompt, tokenizer = prepare_prompt(self.root, 4, 11, 16)
        expected = torch.arange(44).reshape(4, 11).remainder(29).add(3)
        expected[:, 0] = 7
        np.testing.assert_array_equal(prompt, expected.numpy())
        self.assertIsNone(tokenizer)
        path = self.root / "prompt.json"
        write_json(path, prompt.tolist())
        replayed, _ = prepare_prompt(self.root, 4, 1024, 16, prompt_ids=path)
        np.testing.assert_array_equal(replayed, prompt)
        self.assertEqual(replayed.dtype, np.int64)

    def test_text_uses_actual_length_without_padding(self):
        tokenizer = Mock(return_value={"input_ids": [7, 8, 9]})
        with patch("opt_inputs.load_tokenizer", return_value=tokenizer):
            prompt, used = prepare_prompt(self.root, 2, 1024, 16, prompt_text="The capital of France is")
        self.assertIs(used, tokenizer)
        np.testing.assert_array_equal(prompt, [[7, 8, 9], [7, 8, 9]])
        tokenizer.assert_called_once_with("The capital of France is", add_special_tokens=True,
                                          padding=False, truncation=False)

    def test_tokenizer_is_local_only(self):
        factory = Mock()
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=factory)}):
            self.assertIs(load_tokenizer(self.root), factory.from_pretrained.return_value)
        factory.from_pretrained.assert_called_once_with(str(self.root), local_files_only=True,
                                                       use_fast=True, trust_remote_code=False)

    def test_reject_invalid_prompt_files(self):
        path = self.root / "prompt.json"
        for values in ([], [[7, 8], [7]], [[7.0, 8.0]], [[True]], [[7, 1]], [[7, 32]],
                       [[7, -1]], [[7], [7]], [[7] * 64]):
            with self.subTest(values=values):
                write_json(path, values)
                with self.assertRaises(ValueError):
                    prepare_prompt(self.root, 1, 8, 1, prompt_ids=path)

    def test_reject_blank_conflicting_or_overlong_text(self):
        with self.assertRaises(ValueError):
            prepare_prompt(self.root, 1, 8, 1, prompt_text=" ")
        with self.assertRaises(ValueError):
            prepare_prompt(self.root, 1, 8, 1, prompt_ids="unused", prompt_text="hello")
        with patch("opt_inputs.load_tokenizer", return_value=Mock(return_value={"input_ids": [7] * 64})):
            with self.assertRaises(ValueError):
                prepare_prompt(self.root, 1, 8, 1, prompt_text="hello")

    def test_reference_checks_inputs_settings_and_token_shape(self):
        prompt, _ = prepare_prompt(self.root, 1, 8, 2)
        case = comparison_case(self.root, prompt, 2, 16, 8)
        path = self.root / "reference.json"
        write_json(path, dict(case=case, tokens=[[5, 6, 7]]))
        np.testing.assert_array_equal(load_reference(path, case), [[5, 6, 7]])
        for key, value in (("prompt_ids", [[7, 9]]), ("compression_ratio", 4), ("top_r", 8)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_reference(path, dict(case, **{key: value}))
        for record in (dict(tokens=[[5, 6, 7]]), dict(case=case, tokens=[[5]]),
                       dict(case=case, tokens=[[5, 6, 32]])):
            write_json(path, record)
            with self.assertRaises(ValueError):
                load_reference(path, case)

    def test_token_comparison_reports_first_difference(self):
        compare_tokens([[5, 6, 7]], [[5, 6, 7]])
        with self.assertRaisesRegex(ValueError, "batch=0, generated_index=1"):
            compare_tokens([[5, 9, 7]], [[5, 6, 7]])
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare_tokens([[5]], [[5, 6]])

    def test_decode_text_keeps_raw_tokens(self):
        tokenizer = Mock()
        tokenizer.batch_decode.side_effect = [["Paris."], ["The capital of France is Paris."]]
        prompt = np.array([[7, 8, 9]])
        generated = [[10, 11]]
        text = decode_output(tokenizer, prompt, generated)
        self.assertEqual(text["continuation"], ["Paris."])
        self.assertEqual(generated, [[10, 11]])
        self.assertEqual(tokenizer.batch_decode.call_args_list[0].args[0], [[10, 11]])
        self.assertEqual(tokenizer.batch_decode.call_args_list[1].args[0], [[7, 8, 9, 10, 11]])


if __name__ == "__main__":
    unittest.main()
