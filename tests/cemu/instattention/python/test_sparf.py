#!/usr/bin/env python3

import unittest

import numpy as np

from cemu_flexgen import sparf_attention


class SparfAttentionTest(unittest.TestCase):
    def test_full_selection_matches_dense(self):
        random = np.random.default_rng(7)
        query = random.normal(size=(2, 4, 8)).astype(np.float32) / np.sqrt(8)
        keys = random.normal(size=(11, 2, 2, 8)).astype(np.float32)
        values = random.normal(size=keys.shape).astype(np.float32)
        result = sparf_attention(query, keys, values, top_r=8, top_k=11)
        expected = np.empty_like(query)
        for batch in range(2):
            for head in range(4):
                scores = keys[:, batch, head // 2] @ query[batch, head]
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                expected[batch, head] = probability @ values[:, batch, head // 2]
        np.testing.assert_allclose(result.output, expected, rtol=2e-6, atol=2e-6)

    def test_sparse_selection_keeps_current_token(self):
        random = np.random.default_rng(11)
        query = random.normal(size=(1, 3, 16)).astype(np.float16)
        keys = random.normal(size=(17, 1, 3, 16)).astype(np.float16)
        values = random.normal(size=keys.shape).astype(np.float16)
        result = sparf_attention(query, keys, values, top_r=2, top_k=3)
        self.assertTrue(np.all(np.any(result.token_indices == 16, axis=1)))
        self.assertTrue(np.all((result.alpha > 0) & (result.alpha <= 1)))


if __name__ == "__main__":
    unittest.main()
