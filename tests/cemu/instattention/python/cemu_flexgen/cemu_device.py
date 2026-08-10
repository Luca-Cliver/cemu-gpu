from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from _cemu_client import CemuClient, MemoryRange, ProgramTarget
except ImportError as error:
    raise ImportError(
        "cannot import _cemu_client; build it in the Guest with "
        "'make instattention' and add instattention/build-guest to PYTHONPATH"
    ) from error


@dataclass(frozen=True)
class RangeSpec:
    path: str
    size: int
    offset: int = 0

    def __post_init__(self):
        if not self.path:
            raise ValueError("range path is required")
        if self.size <= 0:
            raise ValueError("range size must be positive")
        if self.offset < 0:
            raise ValueError("range offset cannot be negative")
        if self.offset % 512 != 0 or self.size % 512 != 0:
            raise ValueError("range offset and size must be 512-byte aligned")


class CemuDevice:
    def __init__(
        self,
        program_name: str,
        program_path: str,
        function_name: str,
        ranges: Iterable[RangeSpec],
        control_path: str = "/dev/nvme0c3",
        namespace_path: str = "/dev/ng0n3",
        cuda_target: bool = False,
        program_runtime: int = 0,
        runtime_scale_tenths: int = 0,
        indirect: bool = False,
        replace_existing: bool = False,
    ):
        if not program_name:
            raise ValueError("program name is required")
        if not program_path:
            raise ValueError("program path is required")
        if not function_name:
            raise ValueError("function name is required")

        range_specs = tuple(ranges)
        if not range_specs:
            raise ValueError("at least one memory range is required")
        if not all(isinstance(spec, RangeSpec) for spec in range_specs):
            raise TypeError("ranges must contain RangeSpec objects")

        self.program_name = program_name
        self.program_path = program_path
        self.function_name = function_name
        self.ranges = range_specs
        self.control_path = control_path
        self.namespace_path = namespace_path
        self.cuda_target = cuda_target
        self.program_runtime = program_runtime
        self.runtime_scale_tenths = runtime_scale_tenths
        self.indirect = indirect
        self.replace_existing = replace_existing
        self._client: Optional[CemuClient] = None

    @property
    def is_open(self) -> bool:
        return self._client is not None

    @property
    def program_id(self) -> int:
        return self._require_client().program_id

    @property
    def memory_range_set_id(self) -> int:
        return self._require_client().memory_range_set_id

    def open(self):
        if self._client is not None:
            return self

        client = CemuClient(self.control_path, self.namespace_path)
        try:
            target = (
                ProgramTarget.CUDA_DEVICE_POINTER
                if self.cuda_target
                else ProgramTarget.HOST
            )
            client.load_program(
                self.program_name,
                self.program_path,
                self.function_name,
                target,
                self.program_runtime,
                self.runtime_scale_tenths,
                self.indirect,
                self.replace_existing,
            )
            client.activate_program()
            client.create_memory_ranges(
                [MemoryRange(spec.path, spec.offset, spec.size) for spec in self.ranges]
            )
        except Exception:
            client.close()
            raise

        self._client = client
        return self

    def write_tensor(
        self,
        range_index: int,
        value: Any,
        range_offset: int = 0,
    ) -> None:
        spec = self._range_spec(range_index)
        array = self._to_numpy(value)
        self._validate_transfer(spec, array.nbytes, range_offset)
        self._require_client().write_range(range_index, array, range_offset)

    def read_tensor(
        self,
        range_index: int,
        shape: Sequence[int],
        dtype: Any,
        range_offset: int = 0,
    ) -> np.ndarray:
        spec = self._range_spec(range_index)
        normalized_shape = self._normalize_shape(shape)
        normalized_dtype = np.dtype(dtype)
        element_count = int(np.prod(normalized_shape, dtype=np.int64))
        size = element_count * normalized_dtype.itemsize
        self._validate_transfer(spec, size, range_offset)

        raw = self._require_client().read_range(range_index, size, range_offset)
        return np.asarray(raw).view(normalized_dtype).reshape(normalized_shape).copy()

    def execute(
        self,
        cparam1: int = 0,
        cparam2: int = 0,
        group: int = 0,
        chunk_nlb: int = 0,
        runtime: int = 0,
        metadata: Any = None,
    ) -> int:
        if metadata is not None and not isinstance(
            metadata, (bytes, bytearray, memoryview, np.ndarray)
        ):
            metadata = self._to_numpy(metadata)

        return self._require_client().execute(
            cparam1,
            cparam2,
            group,
            chunk_nlb,
            runtime,
            metadata,
        )

    def close(self) -> None:
        if self._client is None:
            return
        self._client.close()
        self._client = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_client(self) -> CemuClient:
        if self._client is None:
            raise RuntimeError("CEMU device is not open")
        return self._client

    def _range_spec(self, range_index: int) -> RangeSpec:
        if not isinstance(range_index, int):
            raise TypeError("range index must be an integer")
        if range_index < 0 or range_index >= len(self.ranges):
            raise IndexError("invalid CEMU memory range index")
        return self.ranges[range_index]

    @staticmethod
    def _validate_transfer(spec: RangeSpec, size: int, range_offset: int) -> None:
        if range_offset < 0 or range_offset % 512 != 0:
            raise ValueError("range offset must be a non-negative multiple of 512")
        if size < 0 or range_offset + size > spec.size:
            raise ValueError("tensor exceeds the configured CEMU memory range")

        aligned_size = (size + 511) // 512 * 512
        if range_offset + aligned_size > spec.size:
            raise ValueError("aligned tensor transfer exceeds the CEMU memory range")

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
        normalized = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized):
            raise ValueError("tensor dimensions cannot be negative")
        return normalized

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value)

        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
            return np.ascontiguousarray(value)

        return np.ascontiguousarray(value)
