import mmap
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from .kv_layout import (
    BLOCK_ALIGNMENT,
    LAYER_ALIGNMENT,
    KvCacheLayout,
    KvChunk,
    align_up,
)


class KvStagingManager:
    def __init__(
        self,
        layout: KvCacheLayout,
        k_cache_path: Any = "/mnt/nvme0/k_cache",
        v_cache_path: Any = "/mnt/nvme0/v_cache",
        k_staging_path: Any = "/mnt/fdm0/k_staging_0",
        v_staging_path: Any = "/mnt/fdm0/v_staging_0",
        staging_bytes: int = 64 * 1024 * 1024,
        replace_existing: bool = False,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if (
            not isinstance(staging_bytes, int)
            or isinstance(staging_bytes, bool)
            or staging_bytes <= 0
            or staging_bytes % BLOCK_ALIGNMENT != 0
        ):
            raise ValueError("staging_bytes must be a positive multiple of 512")

        self.layout = layout
        self.k_cache_path = Path(k_cache_path)
        self.v_cache_path = Path(v_cache_path)
        self.k_staging_path = Path(k_staging_path)
        self.v_staging_path = Path(v_staging_path)
        if self.k_cache_path == self.v_cache_path:
            raise ValueError("K and V cache paths must be different")
        if self.k_staging_path == self.v_staging_path:
            raise ValueError("K and V staging paths must be different")

        self.staging_bytes = staging_bytes
        self.allocation_bytes = align_up(staging_bytes, LAYER_ALIGNMENT)
        self.replace_existing = bool(replace_existing)
        self._k_cache_fd = -1
        self._v_cache_fd = -1
        self._k_staging_fd = -1
        self._v_staging_fd = -1
        self._last_chunk = None

    @property
    def is_open(self) -> bool:
        return all(
            file_descriptor >= 0
            for file_descriptor in (
                self._k_cache_fd,
                self._v_cache_fd,
                self._k_staging_fd,
                self._v_staging_fd,
            )
        )

    @property
    def token_capacity(self) -> int:
        return self.layout.tokens_per_chunk(
            self.staging_bytes,
            self.staging_bytes,
        )

    @property
    def last_chunk(self):
        return self._last_chunk

    def open(self):
        if self.is_open:
            return self

        self._validate_source_file(self.k_cache_path)
        self._validate_source_file(self.v_cache_path)
        self._validate_parent(self.k_staging_path)
        self._validate_parent(self.v_staging_path)

        try:
            source_flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
            self._k_cache_fd = os.open(self.k_cache_path, source_flags)
            self._v_cache_fd = os.open(self.v_cache_path, source_flags)
            self._k_staging_fd = self._open_staging_file(self.k_staging_path)
            self._v_staging_fd = self._open_staging_file(self.v_staging_path)
        except Exception:
            self.close()
            raise
        return self

    def stage_chunk(self, chunk: KvChunk) -> int:
        self._require_open()
        self._validate_chunk(chunk)

        self._copy_file_range_all(
            self._k_cache_fd,
            self._k_staging_fd,
            chunk.copy_size,
            chunk.nvm_offset,
            0,
        )
        self._copy_file_range_all(
            self._v_cache_fd,
            self._v_staging_fd,
            chunk.copy_size,
            chunk.nvm_offset,
            0,
        )
        self._last_chunk = chunk
        return chunk.copy_size

    def read_staged_tokens(
        self,
        token_count: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._require_open()
        if token_count is None:
            if self._last_chunk is None:
                raise RuntimeError("no KV chunk has been staged")
            token_count = self._last_chunk.token_count
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
            or token_count > self.token_capacity
        ):
            raise ValueError("token_count is outside the staging capacity")

        storage_size = token_count * self.layout.token_stride
        key_storage = self._pread_all(self._k_staging_fd, storage_size, 0)
        value_storage = self._pread_all(self._v_staging_fd, storage_size, 0)
        return (
            self._unpack_tokens(key_storage, token_count),
            self._unpack_tokens(value_storage, token_count),
        )

    def record_staged_chunk(self, chunk: KvChunk) -> None:
        self._require_open()
        self._validate_chunk(chunk)
        self._last_chunk = chunk

    def close(self) -> None:
        for attribute_name in (
            "_k_cache_fd",
            "_v_cache_fd",
            "_k_staging_fd",
            "_v_staging_fd",
        ):
            file_descriptor = getattr(self, attribute_name)
            if file_descriptor >= 0:
                os.close(file_descriptor)
                setattr(self, attribute_name, -1)
        self._last_chunk = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open_staging_file(self, path: Path) -> int:
        if self.replace_existing and path.exists():
            path.unlink()

        file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            current_size = os.fstat(file_descriptor).st_size
            if current_size not in (0, self.allocation_bytes):
                raise ValueError(
                    f"{path} has size {current_size}, expected "
                    f"{self.allocation_bytes}"
                )
            if current_size == 0:
                os.posix_fallocate(file_descriptor, 0, self.allocation_bytes)
            return file_descriptor
        except Exception:
            os.close(file_descriptor)
            raise

    def _validate_chunk(self, chunk: KvChunk) -> None:
        if not isinstance(chunk, KvChunk):
            raise TypeError("chunk must be a KvChunk")
        expected_offset = self.layout.token_offset(chunk.layer, chunk.start_token)
        if chunk.token_count <= 0 or chunk.end_token > self.layout.config.max_seq_len:
            raise ValueError("chunk token range is invalid")
        if chunk.nvm_offset != expected_offset:
            raise ValueError("chunk NVM offset does not match the KV layout")
        expected_size = chunk.token_count * self.layout.token_stride
        if chunk.copy_size != expected_size:
            raise ValueError("chunk copy size does not match its token count")
        if chunk.copy_size > self.staging_bytes:
            raise ValueError("chunk does not fit in the staging buffers")

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

    def _require_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("KV staging manager is not open")

    def _validate_source_file(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"KV cache file does not exist: {path}")
        file_size = path.stat().st_size
        if file_size != self.layout.file_size:
            raise ValueError(
                f"{path} has size {file_size}, expected {self.layout.file_size}"
            )

    @staticmethod
    def _validate_parent(path: Path) -> None:
        if not path.parent.is_dir():
            raise FileNotFoundError(f"staging directory does not exist: {path.parent}")

    @staticmethod
    def _copy_file_range_all(
        source_fd: int,
        destination_fd: int,
        size: int,
        source_offset: int,
        destination_offset: int,
    ) -> None:
        copied = 0
        while copied < size:
            result = os.copy_file_range(
                source_fd,
                destination_fd,
                size - copied,
                offset_src=source_offset + copied,
                offset_dst=destination_offset + copied,
            )
            if result <= 0:
                raise OSError("copy_file_range made no progress")
            copied += result

    @staticmethod
    def _pread_all(file_descriptor: int, size: int, offset: int) -> bytes:
        if size <= 0:
            return b""

        with mmap.mmap(-1, size, access=mmap.ACCESS_WRITE) as buffer:
            read_size = 0
            while read_size < size:
                view = memoryview(buffer)[read_size:size]
                try:
                    result = os.preadv(
                        file_descriptor,
                        [view],
                        offset + read_size,
                    )
                finally:
                    view.release()
                if result <= 0:
                    raise EOFError("pread reached the end of the staging file")
                read_size += result
            return buffer[:size]
