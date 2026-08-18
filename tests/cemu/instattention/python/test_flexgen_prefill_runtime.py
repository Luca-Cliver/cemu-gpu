#!/usr/bin/env python3

import argparse
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from flexgen_runtime import run_flexgen_prefill


TEST_DEVICE = torch.device("cpu")


class FlexGenPrefillRuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260818)
        self.batch_size = 2
        self.sequence_length = 5
        self.hidden_size = 8
        self.num_heads = 2
        self.head_dim = self.hidden_size // self.num_heads
        self.inputs = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_size,
            device=TEST_DEVICE,
        )
        self.attention_mask = torch.ones(
            self.batch_size,
            self.sequence_length,
            dtype=torch.bool,
            device=TEST_DEVICE,
        )
        self.weights = [
            torch.randn(
                self.hidden_size,
                self.hidden_size,
                device=TEST_DEVICE,
            )
            for _ in range(4)
        ]
        self.norm_weights = [
            torch.randn(self.hidden_size, device=TEST_DEVICE) for _ in range(2)
        ]

    def test_prefill_returns_flexgen_cache_layout(self):
        result = self._run_prefill(use_rotary_embedding=False)

        self.assertEqual(
            result.hidden_states.shape,
            (self.batch_size, self.sequence_length, self.hidden_size),
        )
        self.assertEqual(
            result.keys.shape,
            (
                self.sequence_length,
                self.batch_size * self.num_heads,
                self.head_dim,
            ),
        )
        self.assertEqual(result.values.shape, result.keys.shape)

        normalized = self._rms_norm(self.norm_weights[0], self.inputs)
        expected_keys = F.linear(normalized, self.weights[1]).view(
            self.batch_size,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        expected_keys = expected_keys.permute(1, 0, 2, 3).reshape_as(result.keys)
        expected_values = F.linear(normalized, self.weights[2]).view(
            self.batch_size,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        expected_values = expected_values.permute(1, 0, 2, 3).reshape_as(
            result.values
        )

        torch.testing.assert_close(result.keys, expected_keys)
        torch.testing.assert_close(result.values, expected_values)

    def test_prefill_is_causal(self):
        original = self._run_prefill(use_rotary_embedding=False)
        changed_inputs = self.inputs.clone()
        changed_inputs[:, -1] += 100
        changed = self._run_prefill(
            inputs=changed_inputs,
            use_rotary_embedding=False,
        )

        torch.testing.assert_close(
            original.hidden_states[:, :-1],
            changed.hidden_states[:, :-1],
        )

    def test_prefill_supports_rope(self):
        result = self._run_prefill(use_rotary_embedding=True)

        self.assertTrue(torch.isfinite(result.hidden_states).all())
        self.assertTrue(torch.isfinite(result.keys).all())
        self.assertTrue(torch.isfinite(result.values).all())

    def test_invalid_attention_mask_is_rejected(self):
        with self.assertRaises(TypeError):
            self._run_prefill(attention_mask=self.attention_mask.float())

    def _run_prefill(self, inputs=None, attention_mask=None, **kwargs):
        return run_flexgen_prefill(
            inputs=self.inputs if inputs is None else inputs,
            attention_mask=(
                self.attention_mask if attention_mask is None else attention_mask
            ),
            query_weight=self.weights[0],
            key_weight=self.weights[1],
            value_weight=self.weights[2],
            output_weight=self.weights[3],
            input_norm_weight=self.norm_weights[0],
            post_attention_norm_weight=self.norm_weights[1],
            num_heads=self.num_heads,
            **kwargs,
        )

    @staticmethod
    def _rms_norm(weight, hidden_states, epsilon=1e-6):
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + epsilon)
        return weight * normalized.to(input_dtype)


def main():
    global TEST_DEVICE

    parser = argparse.ArgumentParser(description="Test the FlexGen Prefill runtime")
    parser.add_argument("--cuda", action="store_true")
    args, unittest_args = parser.parse_known_args()
    if args.cuda:
        if not torch.cuda.is_available():
            parser.error("--cuda requested, but PyTorch cannot access a CUDA device")
        TEST_DEVICE = torch.device("cuda:0")
        print(f"[flexgen-prefill-runtime] device={torch.cuda.get_device_name(0)}")
    else:
        print("[flexgen-prefill-runtime] device=CPU")
    unittest.main(argv=[sys.argv[0], *unittest_args])


if __name__ == "__main__":
    main()
