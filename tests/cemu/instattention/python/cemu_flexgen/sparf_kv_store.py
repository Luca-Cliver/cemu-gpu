import os
import mmap
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from .kv_layout import SparfKvCacheLayout
from phase_profiler import profile_scope


class SparfKvCacheStore:
    """Persist two K organizations and one token-indexed V organization."""

    def __init__(
        self,
        layout: SparfKvCacheLayout,
        token_k_path: Any,
        channel_k_path: Any,
        token_v_path: Any,
        replace_existing: bool = False,
        profiler=None,
    ):
        if not isinstance(layout, SparfKvCacheLayout):
            raise TypeError("layout must be a SparfKvCacheLayout")
        self.layout = layout
        self.token_k_path = Path(token_k_path)
        self.channel_k_path = Path(channel_k_path)
        self.token_v_path = Path(token_v_path)
        if len({self.token_k_path, self.channel_k_path, self.token_v_path}) != 3:
            raise ValueError("SparF cache paths must be different")
        self.replace_existing = bool(replace_existing)
        self.profiler = profiler
        self._token_k_fd = -1
        self._channel_k_fd = -1
        self._token_v_fd = -1
        self._channel_map = None
        shape = (
            layout.config.num_layers,
            layout.config.batch_size,
            layout.config.num_kv_heads,
            layout.config.head_dim,
        )
        self._value_sums = np.zeros(shape, dtype=np.float64)
        self._value_counts = np.zeros(layout.config.num_layers, dtype=np.int64)

    @property
    def is_open(self) -> bool:
        return min(self._token_k_fd, self._channel_k_fd, self._token_v_fd) >= 0

    def open(self):
        if self.is_open:
            return self
        for path in (self.token_k_path, self.channel_k_path, self.token_v_path):
            if not path.parent.is_dir():
                raise FileNotFoundError(f"cache directory does not exist: {path.parent}")
        try:
            self._token_k_fd = self._open(self.token_k_path, self.layout.token_file_size)
            self._channel_k_fd = self._open(self.channel_k_path, self.layout.channel_file_size)
            self._token_v_fd = self._open(self.token_v_path, self.layout.token_file_size)
            self._channel_map = mmap.mmap(self._channel_k_fd, self.layout.channel_file_size)
        except Exception:
            self.close()
            raise
        return self

    def write_tokens(self, layer: int, start_token: int, keys: Any, values: Any) -> None:
        keys = self._normalize("keys", keys, True)
        values = self._normalize("values", values, True)
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        self._validate_span(layer, start_token, keys.shape[0])
        if start_token == 0 and keys.shape[0] > 1:
            token_k = bytearray(self.layout.token_layer_stride)
            token_v = bytearray(self.layout.token_layer_stride)
            channel_k = bytearray(self.layout.channel_layer_stride)
            for batch in range(self.layout.config.batch_size):
                for head in range(self.layout.config.num_kv_heads):
                    token_keys = np.ascontiguousarray(keys[:, batch, head, :])
                    token_values = np.ascontiguousarray(values[:, batch, head, :])
                    token_offset = batch * self.layout.token_batch_stride + head * self.layout.token_head_stride
                    token_k[token_offset:token_offset + token_keys.nbytes] = token_keys.tobytes()
                    token_v[token_offset:token_offset + token_values.nbytes] = token_values.tobytes()
                    channel_keys = np.ascontiguousarray(token_keys.T)
                    head_offset = batch * self.layout.channel_batch_stride + head * self.layout.channel_head_stride
                    for dimension in range(self.layout.config.head_dim):
                        offset = head_offset + dimension * self.layout.channel_stride
                        channel_k[offset:offset + channel_keys[dimension].nbytes] = channel_keys[dimension].tobytes()
            with profile_scope(self.profiler, "prefill.token_k_pwrite", len(token_k)):
                self._pwrite(self._token_k_fd, token_k, layer * self.layout.token_layer_stride)
            with profile_scope(self.profiler, "prefill.token_v_pwrite", len(token_v)):
                self._pwrite(self._token_v_fd, token_v, layer * self.layout.token_layer_stride)
            with profile_scope(self.profiler, "prefill.channel_k_mmap", len(channel_k)):
                self._channel_map[layer * self.layout.channel_layer_stride:(layer + 1) * self.layout.channel_layer_stride] = channel_k
            self._value_sums[layer] = values.astype(np.float64).sum(axis=0)
            self._value_counts[layer] = keys.shape[0]
        else:
            for batch in range(self.layout.config.batch_size):
                for head in range(self.layout.config.num_kv_heads):
                    token_keys = np.ascontiguousarray(keys[:, batch, head, :])
                    token_values = np.ascontiguousarray(values[:, batch, head, :])
                    self._pwrite(self._token_k_fd, token_keys.tobytes(),
                                 self.layout.token_head_offset(layer, batch, head, start_token))
                    self._pwrite(self._token_v_fd, token_values.tobytes(),
                                 self.layout.token_head_offset(layer, batch, head, start_token))
                    view = np.ndarray(
                        (self.layout.config.head_dim, self.layout.config.max_seq_len),
                        dtype=self.layout.config.dtype,
                        buffer=self._channel_map,
                        offset=self.layout.channel_offset(layer, batch, head, 0),
                        strides=(self.layout.channel_stride, self.layout.element_size),
                    )
                    view[:, start_token:start_token + keys.shape[0]] = token_keys.T
            self._value_sums[layer] += values.astype(np.float64).sum(axis=0)
            self._value_counts[layer] += keys.shape[0]

    def write_token(self, layer: int, token: int, key: Any, value: Any) -> None:
        key = self._normalize("key", key, False)
        value = self._normalize("value", value, False)
        self.write_tokens(layer, token, key[None, ...], value[None, ...])

    def read_tokens(self, layer: int, start_token: int, token_count: int) -> Tuple[np.ndarray, np.ndarray]:
        self._require_open()
        self._validate_span(layer, start_token, token_count)
        shape = (
            token_count,
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )
        keys = np.empty(shape, dtype=self.layout.config.dtype)
        values = np.empty_like(keys)
        size = token_count * self.layout.head_bytes
        for batch in range(self.layout.config.batch_size):
            for head in range(self.layout.config.num_kv_heads):
                offset = self.layout.token_head_offset(layer, batch, head, start_token)
                keys[:, batch, head, :] = np.frombuffer(
                    self._pread(self._token_k_fd, size, offset),
                    dtype=self.layout.config.dtype,
                ).reshape(token_count, self.layout.config.head_dim)
                values[:, batch, head, :] = np.frombuffer(
                    self._pread(self._token_v_fd, size, offset),
                    dtype=self.layout.config.dtype,
                ).reshape(token_count, self.layout.config.head_dim)
        return keys, values

    def read_channels(self, layer: int, batch: int, head: int, dimensions, valid_tokens: int) -> np.ndarray:
        self._require_open()
        dimensions = tuple(int(value) for value in dimensions)
        result = np.empty((len(dimensions), valid_tokens), dtype=self.layout.config.dtype)
        size = valid_tokens * self.layout.element_size
        for index, dimension in enumerate(dimensions):
            offset = self.layout.channel_offset(layer, batch, head, dimension)
            result[index] = np.frombuffer(
                self._pread(self._channel_k_fd, size, offset),
                dtype=self.layout.config.dtype,
            )
        return result

    def value_mean(self, layer: int) -> np.ndarray:
        count = int(self._value_counts[layer])
        if count <= 0:
            raise RuntimeError("no values have been written for this layer")
        return np.asarray(
            self._value_sums[layer] / count,
            dtype=self.layout.config.dtype,
        )

    def flush(self) -> None:
        self._require_open()
        self._channel_map.flush()
        for descriptor in (self._token_k_fd, self._channel_k_fd, self._token_v_fd):
            os.fsync(descriptor)

    def close(self) -> None:
        if self._channel_map is not None:
            self._channel_map.flush()
            self._channel_map.close()
            self._channel_map = None
        for name in ("_token_k_fd", "_channel_k_fd", "_token_v_fd"):
            descriptor = getattr(self, name)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, name, -1)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open(self, path: Path, expected_size: int) -> int:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        if self.replace_existing:
            os.ftruncate(descriptor, 0)
        size = os.fstat(descriptor).st_size
        if size not in (0, expected_size):
            os.close(descriptor)
            raise ValueError(f"{path} has size {size}, expected {expected_size}")
        if size == 0:
            os.posix_fallocate(descriptor, 0, expected_size)
        return descriptor

    def _normalize(self, name: str, value: Any, tokens: bool) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.ascontiguousarray(value)
        tail = (
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )
        valid = array.ndim == 4 and array.shape[1:] == tail if tokens else array.shape == tail
        if not valid:
            raise ValueError(f"{name} has invalid shape {array.shape}")
        if array.dtype != self.layout.config.dtype:
            raise TypeError(f"{name} must use {self.layout.config.dtype}")
        return array

    def _validate_span(self, layer: int, token: int, count: int) -> None:
        self.layout.token_head_offset(layer, 0, 0, token)
        if count <= 0 or token + count > self.layout.config.max_seq_len:
            raise ValueError("token span exceeds the configured cache")

    def _require_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("SparF KV cache store is not open")

    @staticmethod
    def _pwrite(descriptor: int, data: bytes, offset: int) -> None:
        view = memoryview(data)
        done = 0
        while done < len(view):
            count = os.pwrite(descriptor, view[done:], offset + done)
            if count <= 0:
                raise OSError("pwrite made no progress")
            done += count

    @staticmethod
    def _pread(descriptor: int, size: int, offset: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = os.pread(descriptor, size - len(data), offset + len(data))
            if not chunk:
                raise EOFError("pread reached end of cache file")
            data.extend(chunk)
        return bytes(data)
