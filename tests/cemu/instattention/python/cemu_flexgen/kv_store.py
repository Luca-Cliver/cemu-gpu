import os
from numbers import Integral
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from .kv_layout import KvCacheLayout


class KvCacheStore:
    def __init__(
        self,
        layout: KvCacheLayout,
        k_path: Any = "/mnt/nvme0/k_cache",
        v_path: Any = "/mnt/nvme0/v_cache",
        replace_existing: bool = False,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")

        self.layout = layout
        self.k_path = Path(k_path)
        self.v_path = Path(v_path)
        if self.k_path == self.v_path:
            raise ValueError("K and V cache paths must be different")

        self.replace_existing = bool(replace_existing)
        self._k_fd = -1
        self._v_fd = -1

    @property
    def is_open(self) -> bool:
        return self._k_fd >= 0 and self._v_fd >= 0

    def open(self):
        if self.is_open:
            return self

        self._validate_parent(self.k_path)
        self._validate_parent(self.v_path)
        try:
            self._k_fd = self._open_cache_file(self.k_path)
            self._v_fd = self._open_cache_file(self.v_path)
        except Exception:
            self.close()
            raise
        return self

    def write_token(
        self,
        layer: int,
        token: int,
        key: Any,
        value: Any,
    ) -> None:
        key_array = self._normalize_tensor("key", key, include_token_axis=False)
        value_array = self._normalize_tensor("value", value, include_token_axis=False)
        self._write_arrays(
            layer,
            token,
            key_array[np.newaxis, ...],
            value_array[np.newaxis, ...],
        )

    def write_tokens(
        self,
        layer: int,
        start_token: int,
        keys: Any,
        values: Any,
    ) -> None:
        key_array = self._normalize_tensor("keys", keys, include_token_axis=True)
        value_array = self._normalize_tensor("values", values, include_token_axis=True)
        if key_array.shape[0] != value_array.shape[0]:
            raise ValueError("keys and values must contain the same number of tokens")
        self._write_arrays(layer, start_token, key_array, value_array)

    def read_token(self, layer: int, token: int) -> Tuple[np.ndarray, np.ndarray]:
        keys, values = self.read_tokens(layer, token, 1)
        return keys[0], values[0]

    def read_tokens(
        self,
        layer: int,
        start_token: int,
        token_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._require_open()
        self._validate_token_span(layer, start_token, token_count)
        offset = self.layout.token_offset(layer, start_token)
        storage_size = int(token_count) * self.layout.token_stride

        key_storage = self._pread_all(self._k_fd, storage_size, offset)
        value_storage = self._pread_all(self._v_fd, storage_size, offset)
        return (
            self._unpack_tokens(key_storage, int(token_count)),
            self._unpack_tokens(value_storage, int(token_count)),
        )

    def flush(self) -> None:
        self._require_open()
        os.fsync(self._k_fd)
        os.fsync(self._v_fd)

    def close(self) -> None:
        if self._k_fd >= 0:
            os.close(self._k_fd)
            self._k_fd = -1
        if self._v_fd >= 0:
            os.close(self._v_fd)
            self._v_fd = -1

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open_cache_file(self, path: Path) -> int:
        file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if self.replace_existing:
                os.ftruncate(file_descriptor, 0)

            current_size = os.fstat(file_descriptor).st_size
            if current_size not in (0, self.layout.file_size):
                raise ValueError(
                    f"{path} has size {current_size}, expected {self.layout.file_size}"
                )
            if current_size == 0:
                os.posix_fallocate(file_descriptor, 0, self.layout.file_size)
            return file_descriptor
        except Exception:
            os.close(file_descriptor)
            raise

    def _write_arrays(
        self,
        layer: int,
        start_token: int,
        key_array: np.ndarray,
        value_array: np.ndarray,
    ) -> None:
        self._require_open()
        token_count = key_array.shape[0]
        if token_count != value_array.shape[0]:
            raise ValueError("K and V token counts do not match")
        self._validate_token_span(layer, start_token, token_count)

        offset = self.layout.token_offset(layer, start_token)
        self._pwrite_all(self._k_fd, self._pack_tokens(key_array), offset)
        self._pwrite_all(self._v_fd, self._pack_tokens(value_array), offset)

    def _normalize_tensor(
        self,
        name: str,
        value: Any,
        include_token_axis: bool,
    ) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.ascontiguousarray(value)
        if array.dtype != self.layout.config.dtype:
            raise TypeError(
                f"{name} dtype {array.dtype} does not match "
                f"layout dtype {self.layout.config.dtype}"
            )

        token_shape = (
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )
        expected_shape = (None,) + token_shape if include_token_axis else token_shape
        if include_token_axis:
            valid_shape = array.ndim == 4 and array.shape[1:] == token_shape
        else:
            valid_shape = array.shape == token_shape
        if not valid_shape:
            raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
        if include_token_axis and array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one token")
        return array

    def _pack_tokens(self, tokens: np.ndarray) -> bytes:
        token_count = tokens.shape[0]
        payload = tokens.view(np.uint8).reshape(token_count, self.layout.token_bytes)
        if self.layout.token_stride == self.layout.token_bytes:
            return payload.tobytes()

        storage = np.zeros(
            (token_count, self.layout.token_stride),
            dtype=np.uint8,
        )
        storage[:, : self.layout.token_bytes] = payload
        return storage.tobytes()

    def _unpack_tokens(self, storage: bytes, token_count: int) -> np.ndarray:
        storage_array = np.frombuffer(storage, dtype=np.uint8).reshape(
            token_count,
            self.layout.token_stride,
        )
        payload = np.ascontiguousarray(
            storage_array[:, : self.layout.token_bytes]
        )
        return payload.view(self.layout.config.dtype).reshape(
            token_count,
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )

    def _validate_token_span(
        self,
        layer: int,
        start_token: int,
        token_count: int,
    ) -> None:
        self.layout.token_offset(layer, start_token)
        if (
            not isinstance(token_count, Integral)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise ValueError("token_count must be a positive integer")
        if start_token + token_count > self.layout.config.max_seq_len:
            raise ValueError("token range exceeds the configured sequence length")

    def _require_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("KV cache store is not open")

    @staticmethod
    def _validate_parent(path: Path) -> None:
        if not path.parent.is_dir():
            raise FileNotFoundError(f"cache directory does not exist: {path.parent}")

    @staticmethod
    def _pwrite_all(file_descriptor: int, data: bytes, offset: int) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            result = os.pwrite(file_descriptor, view[written:], offset + written)
            if result <= 0:
                raise OSError("pwrite made no progress")
            written += result

    @staticmethod
    def _pread_all(file_descriptor: int, size: int, offset: int) -> bytes:
        chunks = []
        read_size = 0
        while read_size < size:
            chunk = os.pread(file_descriptor, size - read_size, offset + read_size)
            if not chunk:
                raise EOFError("pread reached the end of the KV cache file")
            chunks.append(chunk)
            read_size += len(chunk)
        return b"".join(chunks)
