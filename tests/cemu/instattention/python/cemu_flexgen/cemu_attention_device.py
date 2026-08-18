from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .attention_abi import DenseAttentionMetadata
from .cemu_device import CemuDevice, RangeSpec
from .kv_layout import BLOCK_ALIGNMENT, KvCacheLayout, KvChunk
from .kv_staging import KvStagingManager


class AttentionRange(IntEnum):
    QUERY = 0
    KEY_STAGING = 1
    VALUE_STAGING = 2
    SOFTMAX_STATE = 3
    OUTPUT = 4


@dataclass(frozen=True)
class AttentionBufferConfig:
    query_bytes: int
    staging_bytes: int
    state_bytes: int
    output_bytes: int
    query_path: Any = "/mnt/fdm0/attention_query"
    k_staging_path: Any = "/mnt/fdm0/k_staging_0"
    v_staging_path: Any = "/mnt/fdm0/v_staging_0"
    state_path: Any = "/mnt/fdm0/attention_state"
    output_path: Any = "/mnt/fdm0/attention_output"

    def __post_init__(self) -> None:
        for field_name in (
            "query_bytes",
            "staging_bytes",
            "state_bytes",
            "output_bytes",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value % BLOCK_ALIGNMENT != 0
            ):
                raise ValueError(f"{field_name} must be a positive multiple of 512")

        path_names = (
            "query_path",
            "k_staging_path",
            "v_staging_path",
            "state_path",
            "output_path",
        )
        normalized_paths = []
        for field_name in path_names:
            path = Path(getattr(self, field_name))
            if not str(path):
                raise ValueError(f"{field_name} is required")
            object.__setattr__(self, field_name, path)
            normalized_paths.append(path)
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError("attention buffer paths must be different")

    def range_specs(self):
        return (
            RangeSpec(str(self.query_path), self.query_bytes),
            RangeSpec(str(self.k_staging_path), self.staging_bytes),
            RangeSpec(str(self.v_staging_path), self.staging_bytes),
            RangeSpec(str(self.state_path), self.state_bytes),
            RangeSpec(str(self.output_path), self.output_bytes),
        )


class CemuAttentionDevice:
    def __init__(
        self,
        layout: KvCacheLayout,
        buffers: AttentionBufferConfig,
        program_name: str,
        program_path: str,
        function_name: str,
        k_cache_path: Any = "/mnt/nvme0/k_cache",
        v_cache_path: Any = "/mnt/nvme0/v_cache",
        control_path: str = "/dev/nvme0c3",
        namespace_path: str = "/dev/ng0n3",
        cuda_target: bool = False,
        program_runtime: int = 0,
        runtime_scale_tenths: int = 0,
        replace_program: bool = False,
        replace_staging_files: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if not isinstance(buffers, AttentionBufferConfig):
            raise TypeError("buffers must be an AttentionBufferConfig")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.layout = layout
        self.buffers = buffers
        self.logger = logger
        self.ranges = buffers.range_specs()
        self.staging = KvStagingManager(
            layout=layout,
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            k_staging_path=buffers.k_staging_path,
            v_staging_path=buffers.v_staging_path,
            staging_bytes=buffers.staging_bytes,
            replace_existing=replace_staging_files,
        )
        self._device_config = {
            "program_name": program_name,
            "program_path": program_path,
            "function_name": function_name,
            "ranges": self.ranges,
            "control_path": control_path,
            "namespace_path": namespace_path,
            "cuda_target": cuda_target,
            "program_runtime": program_runtime,
            "runtime_scale_tenths": runtime_scale_tenths,
            "indirect": False,
            "replace_existing": replace_program,
        }
        self._device: Optional[CemuDevice] = None

    @property
    def is_open(self) -> bool:
        return self._device is not None and self.staging.is_open

    @property
    def memory_range_set_id(self) -> int:
        return self._require_device().memory_range_set_id

    @property
    def program_id(self) -> int:
        return self._require_device().program_id

    @property
    def memory_range_count(self) -> int:
        return self._require_device().memory_range_count

    @property
    def device(self) -> CemuDevice:
        return self._require_device()

    def open(self):
        if self.is_open:
            return self
        if self._device is not None or self.staging.is_open:
            self.close()

        self._log(
            f"open program={self._device_config['program_name']}, "
            f"cuda_target={self._device_config['cuda_target']}"
        )
        self.staging.open()
        device = CemuDevice(**self._device_config)
        try:
            device.open()
        except Exception:
            device.close()
            self.staging.close()
            raise
        self._device = device
        self._log(
            f"ready program_id={device.program_id}, "
            f"mrs_id={device.memory_range_set_id}, ranges={device.memory_range_count}"
        )
        return self

    def stage_chunk(self, chunk: KvChunk) -> int:
        self._require_device()
        copied = self.staging.stage_chunk(chunk)
        self._log(
            f"stage layer={chunk.layer}, tokens=[{chunk.start_token}, "
            f"{chunk.end_token}), nvm_offset={chunk.nvm_offset}, bytes={copied}"
        )
        return copied

    def run_chunk(
        self,
        query: Any,
        chunk: KvChunk,
        metadata: DenseAttentionMetadata,
        write_query: bool = True,
        read_output: bool = True,
    ) -> Optional[np.ndarray]:
        device = self._require_device()
        if not isinstance(metadata, DenseAttentionMetadata):
            raise TypeError("metadata must be DenseAttentionMetadata")
        metadata.validate_chunk(self.layout, chunk)

        query_array = np.ascontiguousarray(query)
        if query_array.dtype != np.dtype(np.float32):
            raise TypeError("dense Attention query must use float32")
        if query_array.shape != metadata.query_shape:
            raise ValueError(
                f"query shape {query_array.shape} does not match "
                f"{metadata.query_shape}"
            )

        self.stage_chunk(chunk)
        if write_query:
            device.write_tensor(AttentionRange.QUERY, query_array)
        device.execute(metadata=metadata.pack())
        self._log(
            f"execute chunk_tokens={chunk.token_count}, "
            f"reset={metadata.reset_state}, finalize={metadata.finalize}"
        )
        if not read_output:
            return None
        return device.read_tensor(
            AttentionRange.OUTPUT,
            metadata.output_shape,
            np.float32,
        )

    def run_decode(
        self,
        query: Any,
        layer: int,
        valid_tokens: int,
    ) -> np.ndarray:
        """Run one decode query by serially processing all KV chunks."""
        query_array = np.ascontiguousarray(query)
        if query_array.dtype != np.dtype(np.float32):
            raise TypeError("dense Attention query must use float32")
        if query_array.ndim != 3:
            raise ValueError("decode query must have shape [batch, heads, dim]")

        chunks = list(
            self.layout.iter_chunks(
                layer=layer,
                valid_tokens=valid_tokens,
                k_staging_bytes=self.buffers.staging_bytes,
                v_staging_bytes=self.buffers.staging_bytes,
            )
        )
        if not chunks:
            raise ValueError("valid_tokens must produce at least one KV chunk")
        self._log(
            f"decode layer={layer}, valid_tokens={valid_tokens}, chunks={len(chunks)}, "
            f"query={query_array.shape}"
        )

        for chunk_index, chunk in enumerate(chunks):
            metadata = DenseAttentionMetadata.from_layout(
                self.layout,
                num_query_heads=query_array.shape[1],
                token_count=chunk.token_count,
                reset_state=chunk_index == 0,
                finalize=chunk_index == len(chunks) - 1,
            )
            self.run_chunk(
                query_array,
                chunk,
                metadata,
                write_query=chunk_index == 0,
                read_output=False,
            )

        output = self.device.read_tensor(
            AttentionRange.OUTPUT,
            metadata.output_shape,
            np.float32,
        )
        self._log(f"decode complete output={output.shape}")
        return output

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
        self.staging.close()
        self._log("closed")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_device(self) -> CemuDevice:
        if self._device is None:
            raise RuntimeError("CEMU attention device is not open")
        return self._device

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[cemu-attention] {message}")
