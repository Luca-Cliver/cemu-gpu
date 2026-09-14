from pathlib import Path

import numpy as np

from cemu_flexgen import SparfKvCacheStore
from phase_profiler import profile_scope


class SparfAttentionBackend:
    supports_pipelined_decode = False

    def __init__(self, layout, token_k_path, channel_k_path, token_v_path,
                 attention_device, replace_existing=False, logger=None, profiler=None):
        self.layout = layout
        self.paths = tuple(Path(path) for path in (token_k_path, channel_k_path, token_v_path))
        self.attention_device = attention_device
        self.logger = logger
        self.profiler = profiler
        self.store = SparfKvCacheStore(
            layout, *self.paths, replace_existing=replace_existing, profiler=profiler
        )
        self._latest = {}

    @property
    def is_open(self):
        return self.store.is_open and self.attention_device.is_open

    def open(self):
        self.store.open()
        try:
            self.attention_device.open()
        except Exception:
            self.store.close()
            raise
        return self

    def write_prefill(self, layer, keys, values):
        with profile_scope(self.profiler, "prefill.kv_to_cpu_wait_wall"):
            keys, values = self._cache(keys), self._cache(values)
        with profile_scope(self.profiler, "prefill.nvm_write_wall", keys.nbytes + values.nbytes):
            self.store.write_tokens(layer, 0, keys, values)
        return keys.shape[0]

    def append_decode(self, layer, token, key, value):
        with profile_scope(self.profiler, "decode.kv_to_cpu_wait_wall"):
            key, value = self._cache(key), self._cache(value)
        self._latest[layer] = (key[0].copy(), value[0].copy())
        with profile_scope(self.profiler, "decode.nvm_append_wall", key.nbytes + value.nbytes):
            self.store.write_token(layer, token, key[0], value[0])

    def decode(self, layer, query, valid_tokens):
        current_key, current_value = self._latest[layer]
        with profile_scope(self.profiler, "decode.query_to_cpu_wait_wall"):
            query = self._query(query)
        with profile_scope(self.profiler, "decode.value_mean_wall"):
            mean = self.store.value_mean(layer)
        return self.attention_device.run_decode(
            query, layer, valid_tokens, mean,
            current_key, current_value,
            cache_paths=self.paths,
        )

    def flush(self): self.store.flush()

    def close(self):
        self.attention_device.close()
        self.store.close()
        self._latest.clear()

    def __enter__(self): return self.open()
    def __exit__(self, exc_type, exc_value, traceback): self.close()

    def _cache(self, value):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        config = self.layout.config
        if array.ndim == 3:
            array = array.reshape(array.shape[0], config.batch_size,
                                  config.num_kv_heads, config.head_dim)
        return np.ascontiguousarray(array, dtype=config.dtype)

    def _query(self, value):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array.reshape(self.layout.config.batch_size, -1,
                                  self.layout.config.head_dim)
        return np.ascontiguousarray(array, dtype=self.layout.config.dtype)
