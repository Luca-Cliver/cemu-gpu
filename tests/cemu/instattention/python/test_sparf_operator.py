#!/usr/bin/env python3

import argparse
import ctypes
import struct
from pathlib import Path

import numpy as np


class CemuArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("numr", ctypes.c_int), ("mr_addr", ctypes.POINTER(ctypes.c_void_p)),
        ("mr_dev_addr", ctypes.POINTER(ctypes.c_void_p)),
        ("mr_len", ctypes.POINTER(ctypes.c_longlong)),
        ("cparam1", ctypes.c_longlong), ("cparam2", ctypes.c_longlong),
        ("data_buffer", ctypes.c_void_p), ("buffer_len", ctypes.c_longlong),
    ]


def metadata(phase, dtype, batch, heads, dimension, selected):
    code = 2 if dtype == np.dtype(np.float16) else 1
    return struct.pack("<8If", 1, phase, code, batch, heads, dimension, selected, 0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--library")
    args = parser.parse_args()
    dtype = np.dtype(np.float16 if args.cuda else np.float32)
    library_path = args.library or str(
        Path(__file__).resolve().parents[2] / "build" /
        ("sparf_attention_devptr.so" if args.cuda else "sparf_attention.so")
    )
    library = ctypes.CDLL(library_path)
    function = library.sparf_attention
    function.argtypes = [ctypes.POINTER(CemuArgs)]
    function.restype = ctypes.c_longlong
    random = np.random.default_rng(19)
    batch, heads, dimension, selected = 2, 3, 16, 5
    vectors = batch * heads
    query = random.normal(size=(vectors, dimension)).astype(dtype)
    keys = random.normal(size=(vectors, selected, dimension)).astype(dtype)
    values = random.normal(size=(vectors, selected + 1, dimension)).astype(dtype)
    alpha = random.uniform(0.4, 0.95, size=vectors).astype(np.float32)
    probability = np.empty((vectors, selected + 1), np.float32)
    output = np.empty((vectors, dimension), dtype)

    def call(payload, arrays):
        pointers = (ctypes.c_void_p * len(arrays))(*(value.ctypes.data for value in arrays))
        lengths = (ctypes.c_longlong * len(arrays))(*(value.nbytes for value in arrays))
        device_pointers = None
        tensors = None
        if args.cuda:
            import torch
            tensors = [torch.from_numpy(value).cuda() for value in arrays]
            device_pointers = (ctypes.c_void_p * len(arrays))(*(value.data_ptr() for value in tensors))
        packed = ctypes.create_string_buffer(payload)
        arguments = CemuArgs(len(arrays), pointers, device_pointers, lengths, 0, 0,
                             ctypes.addressof(packed), len(payload))
        result = function(ctypes.byref(arguments))
        if tensors is not None and result >= 0:
            for array, tensor in zip(arrays, tensors):
                np.copyto(array, tensor.cpu().numpy())
        return result

    if call(metadata(1, dtype, batch, heads, dimension, selected),
            [query, keys, alpha, probability]) != vectors * dimension:
        raise AssertionError("exact QK phase failed")
    scores = np.einsum("vd,vkd->vk", query.astype(np.float32), keys.astype(np.float32))
    expected_probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    expected_probability /= expected_probability.sum(axis=1, keepdims=True)
    expected_probability *= alpha[:, None]
    expected_probability = np.concatenate((expected_probability, (1 - alpha)[:, None]), axis=1)
    np.testing.assert_allclose(probability, expected_probability, rtol=3e-5, atol=3e-6)
    if call(metadata(2, dtype, batch, heads, dimension, selected),
            [probability, values, output]) != vectors * dimension:
        raise AssertionError("PV phase failed")
    expected = np.einsum("vk,vkd->vd", expected_probability, values.astype(np.float32)).astype(dtype)
    tolerance = 2e-3 if dtype == np.dtype(np.float16) else 2e-5
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
    print(f"[sparf-operator] PASS target={'cuda' if args.cuda else 'cpu'}")


if __name__ == "__main__":
    main()
