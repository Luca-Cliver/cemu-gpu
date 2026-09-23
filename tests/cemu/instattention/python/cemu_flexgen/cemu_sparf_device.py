import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
from _cemu_client import get_file_extents

from phase_profiler import profile_scope

from .cemu_device import CemuDevice, RangeSpec
from .kv_layout import SparfKvCacheLayout, align_up
from .sparf_group_capacity import sparf_attention_range_bytes
from .sparf_workflow_abi import (
    SPARF_WORKFLOW_COMMAND, SPARF_WORKFLOW_TRACE_BYTES, SparfWorkflowTrace,
    pack_sparf_workflow, stage_metadata, stage_context_bytes,
    pack_grouped_sparf_workflow,
    SPARF_STAGES, SPARF_STAGE_NAMES,
)


@dataclass
class SparfStagedDecode:
    """Exclusive lease on a device's FDM ranges until collect or abort."""
    workflow_id: int
    layer: int
    valid_tokens: int
    shape: tuple
    dtype: np.dtype
    metadata: bytes
    completed_stages: int = 0


class CemuSparfDevice:
    def __init__(self, layout, token_k_path, channel_k_path, token_v_path,
                 fdm_directory, program_path, query_heads, top_r,
                 compression_ratio=8, control_path="/dev/nvme0c3",
                 namespace_path="/dev/ng0n3", cuda_target=True,
                 approximate_runtime_ns=0, exact_qk_runtime_ns=0,
                 pv_runtime_ns=0, runtime_model=None, logger=None, name_prefix="sparf", profiler=None, staged=False,
                 buffer_tokens=0, autonomous=False, grouped_store=None, grouped_stream_base=0):
        if not isinstance(layout, SparfKvCacheLayout):
            raise TypeError("layout must be a SparfKvCacheLayout")
        self.layout = layout
        self.paths = tuple(Path(path) for path in (token_k_path, channel_k_path, token_v_path))
        self.query_heads = int(query_heads)
        self.top_r = min(int(top_r), layout.config.head_dim)
        self.compression_ratio = int(compression_ratio)
        if self.top_r <= 0 or self.compression_ratio <= 0:
            raise ValueError("top_r and compression_ratio must be positive")
        self.staged = bool(staged)
        self.autonomous = bool(autonomous)
        if self.staged and self.autonomous:
            raise ValueError("autonomous and Guest-staged modes are mutually exclusive")
        self.buffer_tokens = int(buffer_tokens)
        if self.buffer_tokens < 0:
            raise ValueError("buffer_tokens must be nonnegative")
        self.grouped_store = grouped_store
        self.grouped_stream_base = int(grouped_stream_base)
        if grouped_store is not None:
            streams = layout.config.num_layers * layout.config.batch_size * layout.config.num_kv_heads
            if (not self.autonomous or self.staged or self.buffer_tokens or self.grouped_stream_base < 0 or
                    self.query_heads <= 0 or self.query_heads % layout.config.num_kv_heads or
                    grouped_store.layout.page_bytes != 4096 or
                    grouped_store.layout.head_dim != layout.config.head_dim or
                    grouped_store.layout.dtype != layout.config.dtype or
                    self.grouped_stream_base + streams > grouped_store.buffer_streams or
                    grouped_store.control_path != control_path or grouped_store.namespace_path != namespace_path):
                raise ValueError("grouped storage requires matching geometry, namespace, stream capacity and autonomous mode")
        self._lock = RLock()
        self._active = None
        self._workflow_id = 0
        self.logger = logger
        self.profiler = profiler
        self.runtimes = (approximate_runtime_ns, exact_qk_runtime_ns, pv_runtime_ns)
        self.runtime_model = runtime_model
        vectors = layout.config.batch_size * self.query_heads
        maximum_k = math.ceil(layout.config.max_seq_len / self.compression_ratio)
        root = Path(fdm_directory)
        sizes = sparf_attention_range_bytes(
            batch_size=layout.config.batch_size, query_heads=self.query_heads,
            kv_heads=layout.config.num_kv_heads, head_dim=layout.config.head_dim,
            element_bytes=layout.element_size, max_seq_len=layout.config.max_seq_len,
            top_r=self.top_r, compression_ratio=self.compression_ratio,
            buffer_tokens=self.buffer_tokens,
        )
        names = ("query", "channel_k", "selected_k", "selected_v", "value_mean",
                 "current_k", "current_v", "alpha", "probability", "output", "trace")
        if self.staged:
            sizes += (stage_context_bytes(vectors * maximum_k),)
            names += ("context",)
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
        # Cache files may have been recreated since the previous lifetime.
        self._map_cache.clear()
        if self.grouped_store is None:
            self._maps = self._resolve_maps(self.paths)
        elif not self.grouped_store.registered:
            raise RuntimeError("open grouped storage before its Attention device")
        self._device = CemuDevice(**self._config).open()
        return self

    def _prepare_decode(self, query, layer, valid_tokens, value_mean, current_key=None,
                        current_value=None, cache_paths=None, buffered_kv=None,
                        grouped_stream_base=None):
        if not self.is_open:
            raise RuntimeError("CEMU SparF device is not open")
        if self._active is not None:
            raise RuntimeError("SparF slot already has an active workflow")
        config = self.layout.config
        if (not isinstance(layer, int) or isinstance(layer, bool) or
                not 0 <= layer < config.num_layers):
            raise IndexError("SparF layer is outside the registered layout")
        if (not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool) or
                not 0 < valid_tokens <= config.max_seq_len):
            raise ValueError("valid_tokens is outside the registered layout")
        if self.grouped_store is not None and (not self.grouped_store.registered or
                cache_paths is not None or buffered_kv is not None):
            raise ValueError("grouped Attention uses its registered device storage, not Guest cache paths/tails")
        if self.grouped_store is not None:
            check_ready = getattr(self.grouped_store, "check_ready", None)
            if callable(check_ready):
                check_ready()
        query = np.ascontiguousarray(query, dtype=self.layout.config.dtype)
        expected_query_shape = (config.batch_size, self.query_heads, config.head_dim)
        if query.shape != expected_query_shape:
            raise ValueError(f"query must have shape {expected_query_shape}, got {query.shape}")
        value_mean = np.asarray(value_mean)
        expected_mean_shape = (config.batch_size, config.num_kv_heads, config.head_dim)
        if value_mean.shape != expected_mean_shape:
            raise ValueError(f"value_mean must have shape {expected_mean_shape}, got {value_mean.shape}")
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
        buffered_tokens = None
        request_stream_base = self.grouped_stream_base
        if grouped_stream_base is not None:
            if not isinstance(grouped_stream_base, int) or grouped_stream_base < 0:
                raise ValueError("grouped_stream_base must be a nonnegative integer")
            request_stream_base = grouped_stream_base
        if self.grouped_store is not None:
            local_streams = (self.layout.config.num_layers * self.layout.config.batch_size *
                             self.layout.config.num_kv_heads)
            if request_stream_base + local_streams > self.grouped_store.buffer_streams:
                raise ValueError("grouped Attention stream base exceeds the registered session")
        if buffered_kv is not None:
            keys, values = (np.ascontiguousarray(array, dtype=query.dtype) for array in buffered_kv)
            shape = (self.layout.config.batch_size, self.layout.config.num_kv_heads,
                     self.layout.config.head_dim)
            if (keys.ndim != 4 or keys.shape != values.shape or keys.shape[1:] != shape or
                    keys.shape[0] > self.buffer_tokens or not self.buffer_tokens):
                raise ValueError("buffered KV exceeds the slot capacity or has invalid shape")
            buffered_tokens = len(keys)
            if buffered_tokens:
                current_key, current_value = keys, values
        with profile_scope(self.profiler, "decode.fdm_write_wall"):
            self._device.write_tensor(0, query)
            self._device.write_tensor(4, mean)
            if self.grouped_store is None:
                self._device.write_tensor(5, np.ascontiguousarray(current_key, dtype=query.dtype))
                self._device.write_tensor(6, np.ascontiguousarray(current_value, dtype=query.dtype))
        with profile_scope(self.profiler, "decode.metadata_wall"):
            if self.grouped_store is not None:
                pack_native = getattr(self.grouped_store, "pack_workflow", None)
                if callable(pack_native):
                    metadata = pack_native(
                        self.layout, layer, valid_tokens, self.query_heads, self.top_r, top_k,
                        request_stream_base, approximate_runtime_ns=runtimes[0],
                        exact_qk_runtime_ns=runtimes[1], pv_runtime_ns=runtimes[2], scale=1.0)
                else:
                    metadata = pack_grouped_sparf_workflow(
                        self.layout, layer, valid_tokens, self.query_heads, self.top_r, top_k,
                        self.grouped_store.cookie, request_stream_base,
                        approximate_runtime_ns=runtimes[0], exact_qk_runtime_ns=runtimes[1],
                        pv_runtime_ns=runtimes[2], scale=1.0,
                    )
            else:
                maps = self._resolve_maps(self.paths if cache_paths is None else cache_paths)
                metadata = pack_sparf_workflow(
                    self.layout, layer, valid_tokens, self.query_heads, self.top_r, top_k,
                    *maps, approximate_runtime_ns=runtimes[0],
                    exact_qk_runtime_ns=runtimes[1], pv_runtime_ns=runtimes[2], scale=1.0,
                    buffered_tokens=buffered_tokens,
                    autonomous=self.autonomous,
                )
        self._workflow_id = self._workflow_id % 0xffffffff + 1
        request = SparfStagedDecode(self._workflow_id, layer, valid_tokens,
                                   query.shape, query.dtype, metadata)
        self._active = request
        return request

    def prepare_decode(self, *args, **kwargs):
        """Prepare one slot request without Guest KV selection/read."""
        if not (self.staged or self.autonomous):
            raise RuntimeError("device was not configured for scheduled execution")
        with self._lock:
            return self._prepare_decode(*args, **kwargs)

    def execute_autonomous(self, request):
        """Run one v3 parent command; internal stages belong to the device."""
        with self._lock:
            self._require_request(request)
            if not self.autonomous or request.completed_stages:
                raise RuntimeError("no autonomous SparF request to execute")
            try:
                with profile_scope(self.profiler, "decode.execute_wait_wall"):
                    self._device.execute(cparam1=SPARF_WORKFLOW_COMMAND,
                                         metadata=request.metadata)
                request.completed_stages = len(SPARF_STAGES)
            except Exception:
                self._active = None
                raise

    def execute_stage(self, request):
        """Advance exactly one device stage; another slot can progress separately.

        EXACT_V issues V reads and runs QK before joining the V completion.
        Each call uses the existing blocking ioctl, which releases the GIL.
        """
        with self._lock:
            self._require_request(request)
            if not self.staged or request.completed_stages == len(SPARF_STAGES):
                raise RuntimeError("no SparF stage to execute")
            stage = SPARF_STAGES[request.completed_stages]
            payload = stage_metadata(request.metadata, stage, request.workflow_id)
            try:
                with profile_scope(self.profiler, "decode.execute_wait_wall"):
                    with profile_scope(self.profiler, "decode.stage_" +
                                       SPARF_STAGE_NAMES[request.completed_stages] + "_wall"):
                        self._device.execute(cparam1=SPARF_WORKFLOW_COMMAND, metadata=payload)
                request.completed_stages += 1
            except Exception:
                self._active = None
                raise

    def collect_decode(self, request):
        with self._lock:
            self._require_request(request)
            if request.completed_stages != len(SPARF_STAGES):
                raise RuntimeError("SparF output requested before PV completed")
            try:
                return self._collect_decode(request)
            finally:
                self._active = None

    def abort_decode(self, request):
        # All issued commands have completed before the lock is acquired.
        # The next SELECT overwrites the continuation; no pending FTL jobs exist.
        with self._lock:
            self._require_request(request)
            self._active = None

    def _require_request(self, request):
        if request is not self._active:
            raise RuntimeError("stale request or request belongs to another SparF slot")

    def run_decode(self, *args, **kwargs):
        with self._lock:
            request = self._prepare_decode(*args, **kwargs)
            try:
                if self.staged:
                    for _ in SPARF_STAGES:
                        self.execute_stage(request)
                else:
                    with profile_scope(self.profiler, "decode.execute_wait_wall"):
                        self._device.execute(cparam1=SPARF_WORKFLOW_COMMAND,
                                             metadata=request.metadata)
                    request.completed_stages = len(SPARF_STAGES)
                return self.collect_decode(request)
            finally:
                self._active = None

    def _collect_decode(self, request):
        output_bytes = math.prod(request.shape) * request.dtype.itemsize
        with profile_scope(self.profiler, "decode.output_read_wall", output_bytes):
            output = self._device.read_tensor(9, request.shape, request.dtype)
        with profile_scope(self.profiler, "decode.trace_read_wall", SPARF_WORKFLOW_TRACE_BYTES):
            trace = self._device.read_tensor(10, (SPARF_WORKFLOW_TRACE_BYTES,), np.uint8)
        self.last_trace = SparfWorkflowTrace.unpack(trace)
        if self.profiler is not None:
            for field in ("channel_read_model_ns", "token_k_read_model_ns",
                          "token_v_read_model_ns", "approximate_model_ns",
                          "exact_qk_model_ns", "pv_model_ns", "total_model_ns"):
                self.profiler.record("csd_model." + field, getattr(self.last_trace, field))
            # This is a modeled upper bound: the trace has per-stage target
            # durations, but no timestamps for their actual overlap.  Keep
            # the name explicit so it is not mistaken for measured overlap.
            self.profiler.record("csd_model.qk_v_overlap_upper_bound_ns", min(
                self.last_trace.exact_qk_model_ns, self.last_trace.token_v_read_model_ns))
        self.modeled_runtime_ns_total += self.last_trace.total_model_ns
        self.decode_count += 1
        if self.logger is not None:
            self.logger(f"[cemu-sparf] layer={request.layer}, tokens={request.valid_tokens}, "
                        f"modeled_runtime_ns={self.last_trace.total_model_ns}")
        return output

    def modeled_runtime_summary(self):
        return self.modeled_runtime_ns_total, self.decode_count

    def close(self):
        with self._lock:
            self._active = None
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
