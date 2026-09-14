import math
from pathlib import Path

import numpy as np
from _cemu_client import get_file_extents

from phase_profiler import profile_scope

from .cemu_device import CemuDevice, RangeSpec
from .kv_layout import SparfKvCacheLayout, align_up
from .sparf_workflow_abi import SPARF_WORKFLOW_COMMAND, SPARF_WORKFLOW_TRACE_BYTES, SparfWorkflowTrace, pack_sparf_workflow


class CemuSparfDevice:
    def __init__(self, layout, token_k_path, channel_k_path, token_v_path,
                 fdm_directory, program_path, query_heads, top_r,
                 compression_ratio=8, control_path="/dev/nvme0c3",
                 namespace_path="/dev/ng0n3", cuda_target=True,
                 approximate_runtime_ns=0, exact_qk_runtime_ns=0,
                 pv_runtime_ns=0, runtime_model=None, logger=None, name_prefix="sparf", profiler=None):
        if not isinstance(layout, SparfKvCacheLayout):
            raise TypeError("layout must be a SparfKvCacheLayout")
        self.layout = layout
        self.paths = tuple(Path(path) for path in (token_k_path, channel_k_path, token_v_path))
        self.query_heads = int(query_heads)
        self.top_r = min(int(top_r), layout.config.head_dim)
        self.compression_ratio = int(compression_ratio)
        if self.top_r <= 0 or self.compression_ratio <= 0:
            raise ValueError("top_r and compression_ratio must be positive")
        self.logger = logger
        self.profiler = profiler
        self.runtimes = (approximate_runtime_ns, exact_qk_runtime_ns, pv_runtime_ns)
        self.runtime_model = runtime_model
        vectors = layout.config.batch_size * self.query_heads
        maximum_k = math.ceil(layout.config.max_seq_len / self.compression_ratio)
        element = layout.element_size
        root = Path(fdm_directory)
        sizes = (
            vectors * layout.config.head_dim * element,
            vectors * self.top_r * layout.config.max_seq_len * element,
            vectors * maximum_k * layout.config.head_dim * element,
            vectors * (maximum_k + 1) * layout.config.head_dim * element,
            vectors * layout.config.head_dim * element,
            layout.config.batch_size * layout.config.num_kv_heads * layout.config.head_dim * element,
            layout.config.batch_size * layout.config.num_kv_heads * layout.config.head_dim * element,
            vectors * 4,
            vectors * (maximum_k + 1) * 4,
            vectors * layout.config.head_dim * element,
            SPARF_WORKFLOW_TRACE_BYTES,
        )
        names = ("query", "channel_k", "selected_k", "selected_v", "value_mean",
                 "current_k", "current_v", "alpha", "probability", "output", "trace")
        ranges = tuple(RangeSpec(str(root / f"{name_prefix}_{name}"), align_up(size, 512))
                       for name, size in zip(names, sizes))
        self._config = dict(
            program_name=f"cemu_sparf_{name_prefix}_{root.name}", program_path=str(program_path),
            function_name="sparf_attention", ranges=ranges,
            control_path=control_path, namespace_path=namespace_path,
            cuda_target=cuda_target, replace_existing=True,
        )
        self._device = None
        self._maps = None
        self._map_cache = {}
        self.last_trace = None
        self.modeled_runtime_ns_total = 0
        self.decode_count = 0

    @property
    def is_open(self):
        return self._device is not None

    def open(self):
        if self.is_open:
            return self
        self._maps = self._resolve_maps(self.paths)
        self._device = CemuDevice(**self._config).open()
        return self

    def run_decode(self, query, layer, valid_tokens, value_mean, current_key,
                   current_value, cache_paths=None):
        if not self.is_open:
            raise RuntimeError("CEMU SparF device is not open")
        query = np.ascontiguousarray(query, dtype=self.layout.config.dtype)
        head_indices = np.arange(self.query_heads) // (self.query_heads // self.layout.config.num_kv_heads)
        mean = np.ascontiguousarray(np.asarray(value_mean)[:, head_indices, :], dtype=query.dtype)
        top_k = max(1, math.ceil(valid_tokens / self.compression_ratio))
        runtimes = self.runtimes
        if self.runtime_model is not None:
            estimate = self.runtime_model.estimate(
                self.layout.config.batch_size, self.query_heads,
                self.layout.config.head_dim, valid_tokens, self.top_r, top_k,
                self.layout.element_size,
            )
            runtimes = (estimate.approximate_ns, estimate.exact_qk_ns, estimate.pv_ns)
        with profile_scope(self.profiler, "decode.fdm_write_wall"):
            self._device.write_tensor(0, query)
            self._device.write_tensor(4, mean)
            self._device.write_tensor(5, np.ascontiguousarray(current_key, dtype=query.dtype))
            self._device.write_tensor(6, np.ascontiguousarray(current_value, dtype=query.dtype))
        with profile_scope(self.profiler, "decode.metadata_wall"):
            maps = self._resolve_maps(self.paths if cache_paths is None else cache_paths)
            metadata = pack_sparf_workflow(
                self.layout, layer, valid_tokens, self.query_heads, self.top_r, top_k,
                *maps, approximate_runtime_ns=runtimes[0],
                exact_qk_runtime_ns=runtimes[1], pv_runtime_ns=runtimes[2], scale=1.0,
            )
        with profile_scope(self.profiler, "decode.execute_wait_wall"):
            self._device.execute(cparam1=SPARF_WORKFLOW_COMMAND, metadata=metadata)
        with profile_scope(self.profiler, "decode.output_read_wall", query.nbytes):
            output = self._device.read_tensor(9, query.shape, query.dtype)
        with profile_scope(self.profiler, "decode.trace_read_wall", SPARF_WORKFLOW_TRACE_BYTES):
            trace = self._device.read_tensor(10, (SPARF_WORKFLOW_TRACE_BYTES,), np.uint8)
        self.last_trace = SparfWorkflowTrace.unpack(trace)
        if self.profiler is not None:
            for field in ("channel_read_model_ns", "token_k_read_model_ns",
                          "token_v_read_model_ns", "approximate_model_ns",
                          "exact_qk_model_ns", "pv_model_ns", "total_model_ns"):
                self.profiler.record("csd_model." + field, getattr(self.last_trace, field))
            self.profiler.record("csd_model.qk_v_overlap_ns", min(
                self.last_trace.exact_qk_model_ns, self.last_trace.token_v_read_model_ns))
        self.modeled_runtime_ns_total += self.last_trace.total_model_ns
        self.decode_count += 1
        if self.logger is not None:
            self.logger(f"[cemu-sparf] layer={layer}, tokens={valid_tokens}, r={self.top_r}, k={top_k}, modeled_runtime_ns={self.last_trace.total_model_ns}")
        return output

    def modeled_runtime_summary(self):
        return self.modeled_runtime_ns_total, self.decode_count

    def close(self):
        if self._device is not None:
            self._device.close()
            self._device = None

    def _resolve_maps(self, paths):
        normalized = tuple(Path(path) for path in paths)
        key = tuple(str(path.resolve()) for path in normalized)
        if key not in self._map_cache:
            self._map_cache[key] = (
                get_file_extents(str(normalized[0]), self.layout.token_file_size),
                get_file_extents(str(normalized[1]), self.layout.channel_file_size),
                get_file_extents(str(normalized[2]), self.layout.token_file_size),
            )
        return self._map_cache[key]

    def __enter__(self): return self.open()
    def __exit__(self, exc_type, exc_value, traceback): self.close()
