#!/usr/bin/env python3

import argparse
import ctypes
import math
import struct
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cemu_flexgen.attention_phases_abi import AttentionPhaseMetadata, PV, QK_SOFTMAX
from cemu_flexgen.attention_workflow_abi import (
    ATTENTION_WORKFLOW_COMMAND,
    AttentionWorkflowTrace,
    pack_attention_workflow,
)


OPTIONS = argparse.Namespace(
    library=str(Path(__file__).resolve().parents[2] / "build/attention_phases.so"),
    cuda=False,
)


class CemuArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("numr", ctypes.c_int),
        ("mr_addr", ctypes.POINTER(ctypes.c_void_p)),
        ("mr_dev_addr", ctypes.POINTER(ctypes.c_void_p)),
        ("mr_len", ctypes.POINTER(ctypes.c_longlong)),
        ("cparam1", ctypes.c_longlong),
        ("cparam2", ctypes.c_longlong),
        ("data_buffer", ctypes.c_void_p),
        ("buffer_len", ctypes.c_longlong),
    ]


def make_case(dtype=np.float32, batch=2, query_heads=6, kv_heads=2, dimension=13, tokens=19):
    dtype = np.dtype(dtype)
    payload = batch * kv_heads * dimension * dtype.itemsize
    stride = (payload + 511) // 512 * 512
    config = AttentionPhaseMetadata(
        batch, query_heads, kv_heads, dimension, tokens, stride,
        1.0 / math.sqrt(dimension), dtype,
    )
    random = np.random.default_rng(20260910)
    query = random.normal(size=(batch, query_heads, dimension)).astype(dtype)
    shape = (tokens, batch, kv_heads, dimension)
    keys = random.normal(size=shape).astype(dtype)
    values = random.normal(size=shape).astype(dtype)
    return config, query, keys, values


def padded_tokens(config, values):
    storage = np.zeros((config.token_count, config.token_stride), dtype=np.uint8)
    payload = np.ascontiguousarray(values).view(np.uint8).reshape(config.token_count, -1)
    storage[:, :payload.shape[1]] = payload
    return storage


def reference(config, query, keys, values):
    head_indices = np.arange(config.num_query_heads) // (config.num_query_heads // config.num_kv_heads)
    grouped_keys = keys[:, :, head_indices, :].astype(np.float64)
    scores = np.einsum("bhd,tbhd->bht", query.astype(np.float64), grouped_keys) * config.scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    output = np.einsum("bht,tbhd->bhd", probabilities, values[:, :, head_indices, :].astype(np.float64))
    return probabilities, output.astype(config.dtype)


class PhaseLibrary:
    def __init__(self, path, cuda=False):
        self.cuda = cuda
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.function = self.library.attention_phase
        self.function.argtypes = [ctypes.POINTER(CemuArgs)]
        self.function.restype = ctypes.c_longlong
        if cuda:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.torch = torch

    def call(self, metadata, arrays, lengths=None, alias_output=False):
        payload = ctypes.create_string_buffer(metadata)
        pointers = (ctypes.c_void_p * 3)(*(array.ctypes.data for array in arrays))
        sizes = (ctypes.c_longlong * 3)(*(lengths if lengths is not None else [array.nbytes for array in arrays]))
        device_pointers = None
        if self.cuda:
            tensors = [self.torch.from_numpy(array).cuda() for array in arrays]
            device_pointers = (ctypes.c_void_p * 3)(*(tensor.data_ptr() for tensor in tensors))
        addresses = device_pointers if self.cuda else pointers
        if alias_output:
            addresses[2] = addresses[0]
        arguments = CemuArgs(3, pointers, device_pointers, sizes, 0, 0,
                             ctypes.addressof(payload), len(metadata))
        result = self.function(ctypes.byref(arguments))
        if self.cuda and result >= 0:
            np.copyto(arrays[2], tensors[2].cpu().numpy())
        return result


class AttentionPhasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.operator = PhaseLibrary(OPTIONS.library, OPTIONS.cuda)

    def run_case(self, config, query, keys, values):
        probabilities = np.full((config.batch_size, config.num_query_heads, config.token_count), np.nan, np.float32)
        output = np.full_like(query, np.nan)
        expected_probabilities, expected_output = reference(config, query, keys, values)
        key_storage = padded_tokens(config, keys)
        saved_query = query.copy()
        saved_keys = key_storage.copy()
        np.testing.assert_equal(
            self.operator.call(config.pack(QK_SOFTMAX), [query, key_storage, probabilities]),
            probabilities.size,
        )
        np.testing.assert_allclose(probabilities, expected_probabilities, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(query, saved_query)
        np.testing.assert_array_equal(key_storage, saved_keys)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, rtol=0, atol=2e-6)
        saved = probabilities.copy()
        np.testing.assert_equal(
            self.operator.call(config.pack(PV), [probabilities, padded_tokens(config, values), output]),
            output.size,
        )
        tolerance = 1e-3 if config.dtype == np.float16 else 2e-5
        np.testing.assert_allclose(output, expected_output, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(probabilities, saved)
        return probabilities, output

    def test_mha_gqa_mqa(self):
        for dtype in (np.float32, np.float16):
            for kv_heads in (1, 2, 6):
                for tokens in (1, 19, 257):
                    with self.subTest(dtype=dtype, kv_heads=kv_heads, tokens=tokens):
                        self.run_case(*make_case(dtype, kv_heads=kv_heads, tokens=tokens))

    def test_long_context_and_opt_heads(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                self.run_case(*make_case(dtype, batch=1, query_heads=40, kv_heads=40, dimension=128, tokens=1025))

    def test_stable_softmax(self):
        config, query, keys, values = make_case()
        query.fill(100)
        keys.fill(100)
        self.run_case(config, query, keys, values)

    def test_pv_can_wait_for_different_values(self):
        config, query, keys, values = make_case(np.float16)
        probabilities, _ = self.run_case(config, query, keys, values)
        values *= -2
        output = np.empty_like(query)
        self.assertEqual(self.operator.call(config.pack(PV), [probabilities, padded_tokens(config, values), output]), output.size)
        np.testing.assert_allclose(output, reference(config, query, keys, values)[1], rtol=1e-3, atol=1e-3)

    def test_half_rounding(self):
        config, query, keys, values = make_case(np.float16, batch=1, query_heads=1, kv_heads=1, dimension=7, tokens=2)
        values[0] = np.array([1, -1, 0, 2**-14, 2, 65504, -2**-24], dtype=np.float16)
        values[1] = values[0]
        for dimension in (0, 1, 2, 3, 4, 6):
            values[1, 0, 0, dimension] = np.nextafter(values[0, 0, 0, dimension], np.float16(np.inf))
        query.fill(0)
        _, output = self.run_case(config, query, keys, values)
        np.testing.assert_array_equal(output, reference(config, query, keys, values)[1])

    def test_invalid_metadata_and_ranges(self):
        config, query, keys, _ = make_case()
        output = np.full((config.batch_size, config.num_query_heads, config.token_count), 123, np.float32)
        arrays = [query, padded_tokens(config, keys), output]
        payload = config.pack(QK_SOFTMAX)
        malformed = [b"", payload[:-1], payload + b"\0"]
        for field, value in ((0, 99), (1, 99), (2, 99), (3, 0), (4, 5), (5, 0), (6, 0), (7, 0), (8, 1)):
            changed = bytearray(payload)
            struct.pack_into("<I", changed, field * 4, value)
            malformed.append(bytes(changed))
        for scale in (0, float("nan"), float("inf")):
            malformed.append(payload[:36] + struct.pack("<f", scale))
        malformed.append(struct.pack("<9If", 1, QK_SOFTMAX, 1,
                                     0xFFFFFFFF, 0xFFFFFFFF, 1, 0xFFFFFFFF,
                                     0xFFFFFFFF, 0xFFFFFE00, 1.0))
        for metadata in malformed:
            self.assertEqual(self.operator.call(metadata, arrays), -1)
        for range_index in range(3):
            lengths = [array.nbytes for array in arrays]
            lengths[range_index] -= 1
            self.assertEqual(self.operator.call(payload, arrays, lengths), -1)
        self.assertEqual(self.operator.call(payload, arrays, alias_output=True), -1)
        np.testing.assert_array_equal(output, 123)

    def test_pv_rejects_short_buffers(self):
        config, query, _, values = make_case()
        probabilities = np.ones((config.batch_size, config.num_query_heads, config.token_count), np.float32)
        output = np.full_like(query, 123)
        arrays = [probabilities, padded_tokens(config, values), output]
        for range_index in range(3):
            lengths = [array.nbytes for array in arrays]
            lengths[range_index] -= 1
            self.assertEqual(self.operator.call(config.pack(PV), arrays, lengths), -1)
        np.testing.assert_array_equal(output, 123)

    def test_python_metadata_validation(self):
        config, *_ = make_case()
        self.assertEqual(len(config.pack(PV)), 40)
        for changes in ({"scale": 1e-100}, {"dtype": np.float64}, {"token_stride": 1}, {"num_query_heads": 5}, {"batch_size": True}):
            with self.assertRaises(ValueError):
                replace(config, **changes)
        with self.assertRaises(ValueError):
            config.pack(99)

    def test_workflow_metadata_and_trace(self):
        config, *_ = make_case(tokens=16)
        blocks = config.kv_bytes // 512
        payload = pack_attention_workflow(
            config, [(10, blocks)], [(100, blocks)],
            qk_runtime_ns=200, pv_runtime_ns=100,
        )
        self.assertEqual(ATTENTION_WORKFLOW_COMMAND, 0x43454D5550484153)
        self.assertEqual(len(payload), 72 + 2 * 16)
        trace = AttentionWorkflowTrace.unpack(np.arange(14, dtype=np.uint64))
        self.assertEqual(trace.total_model_ns, 13)
        with self.assertRaises(ValueError):
            pack_attention_workflow(config, [(10, blocks - 1)], [(100, blocks)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default=str(Path(__file__).resolve().parents[2] / "build/attention_phases.so"))
    parser.add_argument("--cuda", action="store_true")
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__] + remaining)
