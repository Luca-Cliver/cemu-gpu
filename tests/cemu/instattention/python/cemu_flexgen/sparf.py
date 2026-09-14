from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SparfResult:
    output: np.ndarray
    channel_indices: np.ndarray
    token_indices: np.ndarray
    alpha: np.ndarray


def sparf_attention(query: Any, keys: Any, values: Any, top_r: int, top_k: int) -> SparfResult:
    """Algorithm 1 from InstAttention for already-scaled decode queries."""
    query = np.asarray(query)
    keys = np.asarray(keys)
    values = np.asarray(values)
    if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("expected Q=[batch,q_heads,dim], K/V=[tokens,batch,kv_heads,dim]")
    tokens, batch, kv_heads, dimension = keys.shape
    if query.shape[0] != batch or query.shape[2] != dimension:
        raise ValueError("query and KV shapes do not match")
    query_heads = query.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if not 0 < top_r <= dimension or not 0 < top_k <= tokens:
        raise ValueError("top_r and top_k must fit the input")

    vectors = batch * query_heads
    channels = np.empty((vectors, top_r), dtype=np.int32)
    selected_tokens = np.empty((vectors, top_k), dtype=np.int32)
    alpha = np.empty(vectors, dtype=np.float32)
    output = np.empty((batch, query_heads, dimension), dtype=np.float32)
    groups = query_heads // kv_heads

    for vector in range(vectors):
        batch_index, head = divmod(vector, query_heads)
        kv_head = head // groups
        q = query[batch_index, head].astype(np.float32)
        channel = np.argpartition(np.abs(q), -top_r)[-top_r:]
        channel = channel[np.argsort(-np.abs(q[channel]), kind="stable")]
        channels[vector] = channel
        selected_q = q[channel]
        ratio = np.abs(q).sum() / max(float(np.abs(selected_q).sum()), np.finfo(np.float32).tiny)
        approximate = keys[:, batch_index, kv_head, :][:, channel].astype(np.float32) @ selected_q
        approximate *= ratio
        approximate -= approximate.max()
        approximate_probability = np.exp(approximate)
        approximate_probability /= approximate_probability.sum()
        chosen = np.argpartition(approximate_probability, -top_k)[-top_k:]
        if tokens - 1 not in chosen:
            chosen[np.argmin(approximate_probability[chosen])] = tokens - 1
        chosen = chosen[np.argsort(-approximate_probability[chosen], kind="stable")]
        selected_tokens[vector] = chosen
        alpha[vector] = approximate_probability[chosen].sum(dtype=np.float32)

        selected_keys = keys[chosen, batch_index, kv_head].astype(np.float32)
        exact = selected_keys @ q
        exact -= exact.max()
        probability = np.exp(exact)
        probability /= probability.sum()
        selected_values = values[chosen, batch_index, kv_head].astype(np.float32)
        value_mean = values[:, batch_index, kv_head].astype(np.float32).mean(axis=0)
        output[batch_index, head] = (
            alpha[vector] * (probability @ selected_values)
            + (1.0 - alpha[vector]) * value_mean
        )

    return SparfResult(
        output=output.astype(query.dtype),
        channel_indices=channels,
        token_indices=selected_tokens,
        alpha=alpha,
    )
