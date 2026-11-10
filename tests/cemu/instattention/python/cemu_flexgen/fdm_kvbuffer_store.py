import secrets
import struct
import json
import os
import time
from numbers import Integral
from pathlib import Path
from threading import RLock, get_native_id

import numpy as np
from phase_profiler import profile_client_call, profile_scope, stack_profiler, profile_lock, request_context

from _cemu_client import CemuClient, MemoryRange

BIND_DESCRIPTOR = struct.Struct("<4I3Q4IQ4I")
BIND_MAGIC = 0x31424B46
BIND_VERSION = 1
APPEND_HEADER = struct.Struct("<4I5Q4IQ2I")
BUFFER_VERSION = 3
BUFFER_HEADER_BYTES = 376
RECEIVE_STATE_BYTES = 152
MAX_DATA_MRS = 126
DATA_MR_BYTES = 2**32 - 65536


class FDM_KVbuffer:
    def __init__(
            self,
            layout,
            batches,
            layers,
            heads,
            max_tokens,
            microbatch_size,
            fdm_directory,
            meta_bytes,
            data_bytes,
            control_path="/dev/nvme0c3",
            namespace_path="/dev/ng0n3",
            nvm_namespace_id=1,
            input_bytes=None,
            batch_pages=256,
            aggregate_program="./build/fdm_kv_aggregate.so",
            profiler=None,
    ):
        dimensions = (batches, layers, heads, microbatch_size, nvm_namespace_id, batch_pages)
        capacities = (max_tokens, meta_bytes, data_bytes)
        for values, limit in ((dimensions, 2**32), (capacities, 2**64)):
            if any(isinstance(value, bool) or not isinstance(value, Integral) or
                   not 0 < value < limit for value in values):
                raise ValueError("FDM KV dimensions and capacities must be positive ABI-sized integers")
        if batches % microbatch_size:
            raise ValueError("batches must be divisible by microbatch_size")
        if max_tokens >= 2**32:
            raise ValueError("FDM layer packets require a uint32 token count")
        if input_bytes is None:
            packet_bytes = 512 + 2 * microbatch_size * heads * max_tokens * layout.head_dim * layout.dtype.itemsize
            input_bytes = max(128 * 1024, (packet_bytes + 511) // 512 * 512)
        if (isinstance(input_bytes, bool) or not isinstance(input_bytes, Integral) or
                not 0 < input_bytes <= 0x7ffff000):
            raise ValueError("FDM input must fit one Linux write request")
        if nvm_namespace_id != 1 or batch_pages > 256:
            raise ValueError("FDM Attention requires NVM namespace 1 and at most 256 pages per batch")
        if meta_bytes >= 2**32 or data_bytes > DATA_MR_BYTES * MAX_DATA_MRS:
            raise ValueError("FDM KV capacity exceeds metadata MR or data MR count limits")
        if any(value % 512 for value in (meta_bytes, data_bytes, input_bytes)):
            raise ValueError("FDM KV ranges and input capacity must be 512-byte aligned")
        if not 512 + 2 * layout.head_dim * layout.dtype.itemsize <= input_bytes <= min(data_bytes, DATA_MR_BYTES):
            raise ValueError("FDM KV input must fit its header and one K/V row within MR1")

        self.layout = layout
        self.profiler = profiler
        self.batches = batches
        self.layers = layers
        self.heads = heads
        self.max_tokens = max_tokens
        self.microbatch_size = microbatch_size
        self.buffer_streams = batches * layers * heads

        self.fdm_directory = Path(fdm_directory)
        self.meta_path = self.fdm_directory / "kv_buffer_meta"
        self.data_path = self.fdm_directory / "kv_buffer_data"
        self.meta_bytes = meta_bytes
        self.data_bytes = data_bytes

        ranges = [MemoryRange(str(self.meta_path), 0, self.meta_bytes)]
        for index, offset in enumerate(range(0, self.data_bytes, DATA_MR_BYTES)):
            path = self.data_path if index == 0 else self.fdm_directory / f"kv_buffer_data_{index}"
            ranges.append(MemoryRange(str(path), 0, min(DATA_MR_BYTES, self.data_bytes - offset)))
        self.ranges = tuple(ranges)
        self.control_path = str(control_path)
        self.namespace_path = str(namespace_path)
        self.session_id = secrets.randbits(64) or 1
        self.generation = 1
        self.nvm_namespace_id = nvm_namespace_id
        self.input_bytes = input_bytes
        self.batch_pages = batch_pages
        self.aggregate_program = str(aggregate_program)

        self.registered = False
        self._client = None
        self._rsid = None
        self._lock = RLock()
        self._failure = None
        self._closed = False
        self._sealed = False
        self._accepted = {}
        self._request_id = 1
        self._append_commands = 0
        self._appended_head_tokens = 0
        self._max_packet = 0
        self._packet_trace = None
        self._payload_layout_phases = set()

    def open(self):
        with self._lock:
            if self.registered:
                return self

            if self._closed or self._failure is not None:
                raise RuntimeError("FDM_KVbuffer is closed or failed") from self._failure

            try:
                if os.getenv("CEMU_PWRITE_PACKET_TRACE"):
                    self._packet_trace = open(os.environ["CEMU_PWRITE_PACKET_TRACE"], "a", buffering=65536)
                self._client = CemuClient(
                    self.control_path,
                    self.namespace_path,
                )
                self._rsid = self._client.create_memory_ranges(self.ranges)
                self._client.load_program(
                    f"fdm_kv_{self.session_id:016x}", self.aggregate_program,
                    "fdm_kv_aggregate",
                )
                self._client.activate_program()
                self._client.bind_fdm_kv_buffer(self._binding_descriptor())
                self.registered = True
            except BaseException as error:
                self._failure = error
                self.close()
                raise
            return self

    def _binding_descriptor(self):
        return BIND_DESCRIPTOR.pack(
            BIND_MAGIC, BIND_VERSION, self.nvm_namespace_id, self.batch_pages,
            self.session_id, self.generation, self.input_bytes,
            self.batches, self.layers, self.heads, self.layout.head_dim,
            self.max_tokens, self.layout.channels_per_page, self.layout.dtype.itemsize,
            self.layout.page_bytes, 0,
        )

    def close(self):
        with self._lock:
            if self._client is not None:
                if self.registered:
                    if self._failure is None and self._accepted and not self._sealed:
                        self.drain()
                    self._client.release_memory_ranges()
                self._client.close()
                self._client = None

            self._rsid = None
            self.registered = False
            self._closed = True
            if self._packet_trace is not None:
                self._packet_trace.close()
                self._packet_trace = None

    def check_ready(self):
        if not self.registered or self._failure is not None:
            raise RuntimeError("FDM KV buffer is closed or failed") from self._failure

    def _coordinate(self, stream):
        per_microbatch = self.layers * self.microbatch_size * self.heads
        microbatch, local = divmod(stream, per_microbatch)
        layer, local = divmod(local, self.microbatch_size * self.heads)
        batch, head = divmod(local, self.heads)
        return microbatch * self.microbatch_size + batch, layer, head

    def _write_packet(self, operation, payload=b"", batch=0, layer=0,
                      head_first=0, head_count=0, token_first=0, token_count=0, phase="drain",
                      batch_count=0):
        if self._request_id >= 2**64 - 1:
            raise OverflowError("FDM packet sequence exhausted")
        wire_bytes = (512 + len(payload) + 511) // 512 * 512
        if wire_bytes > self.input_bytes:
            raise ValueError("FDM KV packet exceeds the input slot")
        with profile_scope(stack_profiler(self.profiler), phase + ".fdm_packet_pack_wall", wire_bytes):
            header = APPEND_HEADER.pack(
                0x31564B46, 2, 512, operation, self.session_id, self.generation,
                self._request_id, wire_bytes, len(payload), batch, layer,
                head_first, head_count, token_first, token_count, batch_count,
            )
            packet = (header.ljust(512, b"\0") + payload).ljust(wire_bytes, b"\0")
        started = time.perf_counter_ns() if self._packet_trace is not None else 0
        success = False
        try:
            profile_client_call(self.profiler, phase + ".fdm_write", self._client.write_range, 1, packet, 0)
            success = True
        finally:
            if self._packet_trace is not None:
                finished = time.perf_counter_ns()
                self._packet_trace.write(json.dumps(dict(version=1, tid=get_native_id(),
                    session=self.session_id, packet_id=self._request_id, phase=phase,
                    bytes=wire_bytes, start_ns=started, end_ns=finished, success=success,
                    request=request_context(self.profiler))) + "\n")
        self._request_id += 1
        self._max_packet = max(self._max_packet, wire_bytes)

    def _pack_payload(self, keys, values, phase, profiler):
        if profiler is None:
            return b"".join(np.ascontiguousarray(array.transpose(1, 0, 2)) for array in (keys, values))
        if phase not in self._payload_layout_phases:
            layouts = {}
            for name, array in (("keys", keys), ("values", values)):
                transposed = array.transpose(1, 0, 2)
                layouts[name] = {
                    "shape": list(array.shape), "strides": list(array.strides),
                    "dtype": str(array.dtype), "bytes": array.nbytes,
                    "c_contiguous": bool(array.flags.c_contiguous),
                    "transposed_shape": list(transposed.shape),
                    "transposed_strides": list(transposed.strides),
                    "transposed_c_contiguous": bool(transposed.flags.c_contiguous),
                }
            profiler.record_metadata_once(phase + ".kv_payload_layout", {
                "numpy_version": np.__version__, "arrays": layouts,
                "packing": "contiguous_buffers_join",
                "copy_metrics": "k/v: contiguous layout preparation; join: serialize both buffers to bytes",
                "sample": "first packet per phase, after ascontiguousarray; strides in bytes",
            })
            self._payload_layout_phases.add(phase)
        started = time.perf_counter_ns()
        key_buffer = np.ascontiguousarray(keys.transpose(1, 0, 2))
        key_finished = time.perf_counter_ns()
        value_buffer = np.ascontiguousarray(values.transpose(1, 0, 2))
        value_finished = time.perf_counter_ns()
        payload = b"".join((key_buffer, value_buffer))
        joined = time.perf_counter_ns()
        del key_buffer, value_buffer
        for name, start, end, byte_count in (
            ("k_copy_wall", started, key_finished, keys.nbytes),
            ("v_copy_wall", key_finished, value_finished, values.nbytes),
            ("join_wall", value_finished, joined, len(payload)),
        ):
            metric = phase + ".kv_payload_pack." + name
            profiler.record(metric, end - start, byte_count)
            profiler.trace_interval(metric, start, end)
        return payload

    def append_batch(self, first_stream, token, keys, values):
        keys, values = (np.ascontiguousarray(array, dtype=self.layout.dtype) for array in (keys, values))
        if (type(first_stream) is not int or type(token) is not int or keys.ndim != 3 or
                keys.shape != values.shape or keys.shape[2] != self.layout.head_dim or
                not keys.shape[0] or not keys.shape[1] or first_stream < 0 or
                first_stream + keys.shape[1] > self.buffer_streams or token < 0 or
                token + len(keys) > self.max_tokens):
            raise ValueError("FDM APPEND requires bounded [tokens, heads, head_dim] K/V")
        with profile_lock(self.profiler, "prefill.store" if token == 0 else "decode.store", self._lock):
            self.check_ready()
            if self._sealed or any(self._accepted.get(head, 0) != token
                                  for head in range(first_stream, first_stream + keys.shape[1])):
                raise ValueError("FDM APPEND must extend each head's accepted prefix")
            batch, layer, head = self._coordinate(first_stream)
            columns = keys.shape[1]
            if columns <= self.heads - head:
                batch_count, head_count = 1, columns
            elif head == 0 and columns % self.heads == 0:
                batch_count, head_count = columns // self.heads, self.heads
            else:
                raise ValueError("FDM layer packet must contain a rectangular batch/head range")
            if batch_count > self.microbatch_size - batch % self.microbatch_size:
                raise ValueError("FDM layer packet cannot cross a layer/microbatch boundary")
            payload_bytes = keys.nbytes + values.nbytes
            if (512 + payload_bytes + 511) // 512 * 512 > self.input_bytes:
                raise ValueError("FDM layer packet exceeds input capacity; increase input_bytes")
            phase = "prefill" if token == 0 else "decode"
            try:
                detailed = stack_profiler(self.profiler)
                with profile_scope(detailed, phase + ".kv_payload_pack_wall", payload_bytes):
                    payload = self._pack_payload(keys, values, phase, detailed)
                self._write_packet(1, payload, batch, layer, head, head_count, token, len(keys),
                                   phase, batch_count=batch_count)
                self._append_commands += 1
                self._appended_head_tokens += columns * len(keys)
                for stream in range(first_stream, first_stream + columns):
                    self._accepted[stream] = token + len(keys)
            except BaseException as error:
                self._failure = error
                raise

    def pack_workflow(self, layout, layer, tokens, query_heads, top_r, top_k, stream_base, **kwargs):
        from cemu_flexgen.sparf_workflow_abi import pack_grouped_sparf_workflow

        self.check_ready()
        per_microbatch = self.layers * self.microbatch_size * self.heads
        if (type(stream_base) is not int or stream_base < 0 or
                stream_base + per_microbatch > self.buffer_streams or stream_base % per_microbatch):
            raise ValueError("FDM Attention must start at a complete microbatch")
        metadata = bytearray(pack_grouped_sparf_workflow(
            layout, layer, tokens, query_heads, top_r, top_k, 1, 0, **kwargs)[:-16])
        flags = struct.unpack_from("<I", metadata, 4)[0]
        struct.pack_into("<I", metadata, 4, flags | 16)
        metadata.extend(struct.pack("<2Q2I", self.session_id, self.generation,
            stream_base // per_microbatch * self.microbatch_size, self.nvm_namespace_id))
        return bytes(metadata)

    def pack_batch_workflow(self, layout, layer, query_heads, top_r, rows, *, scale=1.0,
                            runtimes=(0, 0, 0), trace=False):
        """One FDM EXEC for reordered, variable-length decode sequences."""
        from cemu_flexgen.sparf_workflow_abi import _HEADER

        self.check_ready()
        config = layout.config
        if (not 0 <= layer < self.layers or not rows or len(rows) > config.batch_size or
                config.num_layers != self.layers or config.num_kv_heads != self.heads or
                config.max_seq_len > self.max_tokens or
                config.head_dim != self.layout.head_dim or config.dtype != self.layout.dtype or
                not 0 < query_heads or query_heads % self.heads or
                not 0 < top_r <= config.head_dim or not np.isfinite(scale) or scale <= 0):
            raise ValueError("invalid ragged Attention geometry")
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("duplicate KV slot in Attention batch")
        if len(runtimes) != 3 or any(type(value) is not int or not 0 <= value < 2**32
                                     for value in runtimes):
            raise ValueError("three uint32 modeled runtimes are required")
        history, selected, payload = 0, 0, bytearray()
        for slot, length, top_k in rows:
            if (any(type(x) is not int for x in (slot, length, top_k)) or
                    not 0 <= slot < self.batches or not 0 < length <= config.max_seq_len or
                    not 0 < top_k <= length):
                raise ValueError("invalid Attention batch row")
            payload.extend(struct.pack("<5I", slot, length, top_k, history, selected))
            history += length
            selected += top_k
        header = _HEADER.pack(
            3, 4 | 16 | 32 | int(bool(trace)), *runtimes, 0, 0, 0,
            2 if config.dtype == np.dtype(np.float16) else 1,
            len(rows), query_heads, self.heads, config.head_dim, config.max_seq_len,
            max(r[1] for r in rows), layer, top_r, max(r[2] for r in rows),
            0, 0, 0, 0, 0, 0, 0, float(scale), 0, 0)
        return header + struct.pack("<2Q2I", self.session_id, self.generation,
                                    0, self.nvm_namespace_id) + payload

    def flush(self):
        self.check_ready()

    def drain(self):
        with self._lock:
            self.check_ready()
            if not self._sealed:
                try:
                    self._write_packet(2)
                    self._sealed = True
                except BaseException as error:
                    self._failure = error
                    raise

    def statistics(self):
        with self._lock:
            self.check_ready()
            header = self._client.read_range_bytes(0, 512, 0)
            version, size = struct.unpack_from("<2I", header)
            if (version, size) != (BUFFER_VERSION, BUFFER_HEADER_BYTES):
                raise RuntimeError("unexpected FDM KV metadata ABI")
            stats = dict(zip(("append_commands", "appended_head_tokens", "staged_pages", "write_bytes",
                              "read_bytes", "write_batches", "write_blocks", "partial_blocks",
                              "inflight_blocks", "peak_inflight_blocks", "pending_ready_ns"),
                             struct.unpack_from("<11Q", header, 264)))
            sealed, failed = struct.unpack_from("<2I", header, 184)
            bitmap_bytes = (self.input_bytes // 4 + 7) // 8
            head_offset = (size + RECEIVE_STATE_BYTES + bitmap_bytes + 7) // 8 * 8
            aligned = head_offset // 512 * 512
            head = self._client.read_range_bytes(0, (head_offset - aligned + 40 + 511) // 512 * 512, aligned)
            stats.update(zip(("accepted_stream0", "token_k", "channel_k", "token_v"),
                             struct.unpack_from("<4Q", head, head_offset - aligned)))
            stats.update(sealed=sealed, failed=failed, memory_bytes=self.meta_bytes + self.data_bytes,
                         client_append_commands=self._append_commands,
                         client_appended_head_tokens=self._appended_head_tokens,
                         client_append_batch_max_bytes=self._max_packet,
                         client_append_queue_depth=1, client_append_peak_inflight=int(bool(self._append_commands)))
            return stats

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
