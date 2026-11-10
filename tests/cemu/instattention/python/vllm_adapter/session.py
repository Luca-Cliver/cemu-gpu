"""Request identity and device resources, independent of vLLM and torch."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from phase_profiler import profile_scope


@dataclass(frozen=True)
class Config:
    fdm_directory: str
    program_path: str
    aggregate_program: str
    request_capacity: int = 1
    max_tokens: int = 2048
    attention_slots: int = 1
    top_r: int = 16
    compression_ratio: int = 8
    channels_per_page: int = 8
    meta_bytes: int = 64 * 1024**2
    data_bytes: int = 256 * 1024**2
    batch_pages: int = 256
    control_path: str = "/dev/nvme0c3"
    namespace_path: str = "/dev/ng0n3"
    runtime_config: str | None = None
    softmax_ratio_file: str | None = None
    read_trace: bool = False
    fixed_batch_size: int | None = None

    def __post_init__(self):
        for name in ("request_capacity", "max_tokens", "attention_slots", "top_r",
                     "compression_ratio", "channels_per_page", "meta_bytes",
                     "data_bytes", "batch_pages"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.attention_slots != 1 or self.max_tokens > 65536:
            raise ValueError("requires one batched Attention slot and max_tokens <=65536")
        if bool(self.runtime_config) != bool(self.softmax_ratio_file):
            raise ValueError("runtime_config and softmax_ratio_file must be supplied together")
        if self.fixed_batch_size is not None and (
                type(self.fixed_batch_size) is not int or
                not 1 <= self.fixed_batch_size <= self.request_capacity):
            raise ValueError("fixed_batch_size must be a positive integer within request_capacity")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Work:
    request_id: str
    slot: int
    offset: int
    count: int
    start: int
    prefill: bool

    @property
    def end(self):
        return self.start + self.count


class Session:
    """One FDM binding. Slots are never reused until the session is closed.

    append() publishes the prefix/mean only after APPEND succeeds. Each layer
    submits one ragged decode batch synchronously; no per-sequence job queues.
    """

    def __init__(self, config, layers, query_heads, kv_heads, head_dim, scale,
                 *, store=None, device=None, profiler=None):
        self.config = config
        self.layers, self.query_heads = layers, query_heads
        self.kv_heads, self.head_dim, self.scale = kv_heads, head_dim, scale
        if min(layers, query_heads, kv_heads, head_dim) <= 0 or query_heads % kv_heads:
            raise ValueError("invalid model geometry")
        self.store, self.device = store, device
        self.profiler = profiler
        self.step = 0
        self.runtime_model = load_runtime(config)
        self._stack = ExitStack()
        self._requests = {}
        self._next_slot = 0
        self._counts = {}
        self._sums = {}
        self._means = np.empty((config.request_capacity, kv_heads, head_dim), dtype=np.float16)
        self.closed = False
        self.failure = None
        self.final_statistics = None
        self._fdm_directory = None
        self._fdm_files = ()

    def check_ready(self):
        if self.closed or self.failure is not None:
            raise RuntimeError("CEMU session is closed or failed; start a new session") from self.failure

    def finish(self, request_ids):
        for request_id in request_ids:
            slot = self._requests.pop(request_id, None)
            if slot is not None:
                for layer in range(self.layers):
                    self._counts.pop((slot, layer), None)
                    self._sums.pop((slot, layer), None)

    def plan(self, request_ids, starts, prompts, counts):
        self.check_ready()
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request ID in scheduled batch")
        if not len(request_ids) == len(starts) == len(prompts) == len(counts):
            raise ValueError("inconsistent scheduled batch dimensions")
        if self.config.fixed_batch_size is not None and (
                len(request_ids) != self.config.fixed_batch_size or
                len(set(zip(starts, prompts, counts))) != 1):
            raise ValueError(
                f"fixed-batch experiment requires B{self.config.fixed_batch_size} with equal token ranges; "
                f"scheduled B{len(request_ids)}, starts={list(starts)}, counts={list(counts)}; "
                "enqueue all prompts behind the admission barrier before resuming the scheduler")
        work, offset = [], 0
        for request_id, start, prompt, count in zip(request_ids, starts, prompts, counts):
            start, prompt, count = int(start), int(prompt), int(count)
            prefill = start < prompt
            if count <= 0 or start < 0 or start + count > self.config.max_tokens:
                raise ValueError("scheduled token range exceeds CEMU capacity")
            if (prefill and (start != 0 or count != prompt)) or (not prefill and count != 1):
                raise ValueError("CEMU requires full prefill and one-token decode")
            if request_id not in self._requests:
                if not prefill:
                    raise ValueError("decode request has no CEMU KV prefix")
                if self._next_slot == self.config.request_capacity:
                    raise RuntimeError("CEMU request capacity exhausted: device slots cannot yet be recycled")
                self._requests[request_id] = self._next_slot
                self._next_slot += 1
            slot = self._requests[request_id]
            for layer in range(self.layers):
                if self._counts.get((slot, layer), 0) != start:
                    raise ValueError("CEMU prefix mismatch: replay, preemption or incomplete prior step")
            work.append(Work(request_id, slot, offset, count, start, prefill))
            offset += count
        self.step += 1
        return tuple(work)

    def profile_context(self, work, layer):
        return dict(framework="vllm", id=f"vllm:{self.step}:{layer}", step=self.step,
                    layer=layer, read_trace=self.config.read_trace,
                    rows=[dict(request_id=item.request_id, slot=item.slot, start=item.start,
                               count=item.count, prefill=item.prefill) for item in work])

    def open(self):
        self.check_ready()
        if self.store is not None:
            return self
        from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
        from cemu_flexgen.fdm_kvbuffer_store import FDM_KVbuffer
        from cemu_flexgen.kv_layout import KvLayoutConfig, SparfKvCacheLayout
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout
        from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes

        c = self.config
        grouped_attention_reserve_bytes(
            attention_slots=1, batch_size=c.request_capacity,
            query_heads=self.query_heads, kv_heads=self.kv_heads,
            head_dim=self.head_dim, element_bytes=2, max_seq_len=c.max_tokens,
            top_r=c.top_r, compression_ratio=c.compression_ratio)
        directory = Path(c.fdm_directory)
        directory.mkdir(parents=True, exist_ok=False)
        self._fdm_directory = directory
        try:
            self.store = self._stack.enter_context(FDM_KVbuffer(
                SparfGroupLayout(self.head_dim, c.channels_per_page),
                c.request_capacity, self.layers, self.kv_heads, c.max_tokens, 1,
                c.fdm_directory, c.meta_bytes, c.data_bytes,
                control_path=c.control_path, namespace_path=c.namespace_path,
                aggregate_program=c.aggregate_program, batch_pages=c.batch_pages,
                profiler=self.profiler))
            self._fdm_files = tuple(Path(region.path) for region in self.store.ranges)
            layout = SparfKvCacheLayout(KvLayoutConfig(
                self.layers, c.max_tokens, c.request_capacity, self.kv_heads, self.head_dim, np.float16))
            self.device = self._stack.enter_context(CemuSparfDevice(
                layout, "unused", "unused", "unused", c.fdm_directory, c.program_path,
                self.query_heads, c.top_r, compression_ratio=c.compression_ratio,
                control_path=c.control_path, namespace_path=c.namespace_path,
                name_prefix="vllm_batch", autonomous=True, grouped_store=self.store,
                attention_scale=self.scale, read_trace=c.read_trace, ragged=True,
                runtime_model=self.runtime_model, profiler=self.profiler))
            self._fdm_files += tuple(Path(region.path) for region in self.device._config["ranges"])
        except BaseException as error:
            self.failure = error
            self._stack.close()
            raise
        return self

    def stream_base(self, work):
        return work.slot * self.layers * self.kv_heads

    def append(self, work, layer, keys, values):
        self.check_ready()
        if self._requests.get(work.request_id) != work.slot:
            raise ValueError("stale CEMU request identity")
        if not 0 <= layer < self.layers:
            raise ValueError("layer outside CEMU geometry")
        identity = (work.slot, layer)
        if self._counts.get(identity, 0) != work.start:
            raise ValueError("duplicate or noncontiguous CEMU append")
        shape = (work.count, self.kv_heads, self.head_dim)
        if keys.shape != shape or values.shape != shape or keys.dtype != np.float16 or values.dtype != np.float16:
            raise ValueError("CEMU expects FP16 K/V [tokens,kv_heads,head_dim]")
        try:
            phase = "prefill" if work.prefill else "decode"
            if self.store is None:
                with profile_scope(self.profiler, "session.open_wall"):
                    self.open()
            with profile_scope(self.profiler, phase + ".append_wall"):
                self.store.append_batch(self.stream_base(work) + layer * self.kv_heads,
                                        work.start, keys, values)
            with profile_scope(self.profiler, phase + ".value_sum_wall"):
                if identity not in self._sums:
                    self._sums[identity] = values.sum(axis=0, dtype=np.float64)
                else:
                    total = self._sums[identity]
                    increment = values[0] if work.count == 1 else values.sum(axis=0, dtype=np.float64)
                    np.add(total, increment, out=total)
            self._counts[identity] = work.end
        except BaseException as error:
            self.failure = error
            raise

    def decode_batch(self, work, layer, query, *, output=None):
        """Preserve vLLM row order, passing stable KV slots and lengths to CEMU."""
        self.check_ready()
        if not work or len(work) > self.config.request_capacity or len({w.slot for w in work}) != len(work):
            raise ValueError("empty, oversized or duplicate decode batch")
        with profile_scope(self.profiler, "decode.mean_and_rows_wall"):
            means = self._means[:len(work)]
            rows = []
            for i, w in enumerate(work):
                identity = (w.slot, layer)
                if (w.prefill or self._requests.get(w.request_id) != w.slot or
                        self._counts.get(identity) != w.end):
                    raise ValueError("decode requires accepted current KV and live request identity")
                np.divide(self._sums[identity], w.end, out=means[i], casting="unsafe")
                top_k = (w.end + self.config.compression_ratio - 1) // self.config.compression_ratio
                rows.append((w.slot, w.end, top_k))
        try:
            options = {} if output is None else {"output": output}
            with profile_scope(self.profiler, "decode.batch_wait_wall"):
                return self.device.run_decode(query, layer, max(w.end for w in work), means,
                                              batch_rows=rows, **options)
        except BaseException as error:
            self.failure = error
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self.device is not None:
                    self.device.close()
                if self.store is not None and self.failure is None:
                    self.store.drain()
                    self.final_statistics = self.store.statistics()
                    if not self.final_statistics["sealed"] or self.final_statistics["failed"]:
                        raise RuntimeError("CEMU terminal drain did not seal successfully")
            finally:
                self._stack.close()
            if self.failure is None and self._fdm_directory is not None:
                for path in self._fdm_files:
                    path.unlink(missing_ok=True)
                self._fdm_directory.rmdir()
                self._fdm_directory = None
                self._fdm_files = ()


def load_runtime(config):
    if config.runtime_config is None:
        return None
    from experiments.config import load_experiment_config
    from experiments.runtime_model import SparfAttentionRuntimeModel
    from experiments.softmax_ratio import SoftmaxRatioTable

    return SparfAttentionRuntimeModel(
        load_experiment_config(config.runtime_config).instcsd,
        SoftmaxRatioTable.load(config.softmax_ratio_file))
