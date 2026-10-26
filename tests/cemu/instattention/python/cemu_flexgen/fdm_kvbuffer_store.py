import secrets
import struct
from numbers import Integral
from pathlib import Path
from threading import RLock

import numpy as np

from _cemu_client import CemuClient, MemoryRange

BIND_DESCRIPTOR = struct.Struct("<4I3Q4IQ4I")
BIND_MAGIC = 0x31424B46
BIND_VERSION = 1
APPEND_HEADER = struct.Struct("<4I5Q4IQ2I")


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
            input_bytes=128 * 1024,
            batch_pages=256,
            aggregate_program="./build/fdm_kv_aggregate.so",
    ):
        dimensions = (batches, layers, heads, microbatch_size, nvm_namespace_id, batch_pages)
        capacities = (max_tokens, meta_bytes, data_bytes, input_bytes)
        for values, limit in ((dimensions, 2**32), (capacities, 2**64)):
            if any(isinstance(value, bool) or not isinstance(value, Integral) or
                   not 0 < value < limit for value in values):
                raise ValueError("FDM KV dimensions and capacities must be positive ABI-sized integers")
        if batches % microbatch_size:
            raise ValueError("batches must be divisible by microbatch_size")
        if nvm_namespace_id != 1 or batch_pages > 256:
            raise ValueError("FDM Attention requires NVM namespace 1 and at most 256 pages per batch")
        if meta_bytes >= 2**32 or data_bytes >= 2**32:
            raise ValueError("each CEMU MR length must fit uint32; split larger buffers first")
        if any(value % 512 for value in (meta_bytes, data_bytes, input_bytes)):
            raise ValueError("FDM KV ranges and input capacity must be 512-byte aligned")
        if not 512 + 2 * layout.head_dim * layout.dtype.itemsize <= input_bytes <= data_bytes:
            raise ValueError("FDM KV input must fit its header and one K/V row within MR1")

        self.layout = layout
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

        self.ranges = (MemoryRange(str(self.meta_path), 0, self.meta_bytes),
                       MemoryRange(str(self.data_path), 0, self.data_bytes))
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

    def open(self):
        with self._lock:
            if self.registered:
                return self

            if self._closed or self._failure is not None:
                raise RuntimeError("FDM_KVbuffer is closed or failed") from self._failure

            try:
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
                      head_first=0, head_count=0, token_first=0, token_count=0):
        if self._request_id >= 2**64 - 1:
            raise OverflowError("FDM packet sequence exhausted")
        wire_bytes = (512 + len(payload) + 511) // 512 * 512
        if wire_bytes > self.input_bytes:
            raise ValueError("FDM KV packet exceeds the input slot")
        header = APPEND_HEADER.pack(
            0x31564B46, 1, 512, operation, self.session_id, self.generation,
            self._request_id, wire_bytes, len(payload), batch, layer,
            head_first, head_count, token_first, token_count, 0,
        )
        packet = (header.ljust(512, b"\0") + payload).ljust(wire_bytes, b"\0")
        self._client.write_range(1, packet, 0)
        self._request_id += 1
        self._max_packet = max(self._max_packet, wire_bytes)

    def append_batch(self, first_stream, token, keys, values):
        keys, values = (np.ascontiguousarray(array, dtype=self.layout.dtype) for array in (keys, values))
        if (type(first_stream) is not int or type(token) is not int or keys.ndim != 3 or
                keys.shape != values.shape or keys.shape[2] != self.layout.head_dim or
                not keys.shape[0] or not keys.shape[1] or first_stream < 0 or
                first_stream + keys.shape[1] > self.buffer_streams or token < 0 or
                token + len(keys) > self.max_tokens):
            raise ValueError("FDM APPEND requires bounded [tokens, heads, head_dim] K/V")
        with self._lock:
            self.check_ready()
            if self._sealed or any(self._accepted.get(head, 0) != token
                                  for head in range(first_stream, first_stream + keys.shape[1])):
                raise ValueError("FDM APPEND must extend each head's accepted prefix")
            row_bytes = 2 * self.layout.head_dim * self.layout.dtype.itemsize
            payload_limit = self.input_bytes - 512
            tile = min(2 * self.layout.token_group_tokens, payload_limit // row_bytes)
            try:
                for start in range(0, len(keys), tile):
                    count = min(tile, len(keys) - start)
                    column = 0
                    while column < keys.shape[1]:
                        batch, layer, head = self._coordinate(first_stream + column)
                        heads = min(self.heads - head, keys.shape[1] - column,
                                    payload_limit // (count * row_bytes))
                        payload = b"".join(array[start:start + count, column:column + heads]
                                           .transpose(1, 0, 2).tobytes() for array in (keys, values))
                        self._write_packet(1, payload, batch, layer, head, heads, token + start, count)
                        self._append_commands += 1
                        self._appended_head_tokens += heads * count
                        for stream in range(first_stream + column, first_stream + column + heads):
                            self._accepted[stream] = token + start + count
                        column += heads
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
            if (version, size) != (1, 360):
                raise RuntimeError("unexpected FDM KV metadata ABI")
            stats = dict(zip(("append_commands", "appended_head_tokens", "staged_pages", "write_bytes",
                              "read_bytes", "write_batches", "write_blocks", "partial_blocks",
                              "inflight_blocks", "peak_inflight_blocks", "pending_ready_ns"),
                             struct.unpack_from("<11Q", header, 264)))
            sealed, failed = struct.unpack_from("<2I", header, 184)
            bitmap_bytes = (self.input_bytes // 4 + 7) // 8
            head_offset = (360 + 128 + bitmap_bytes + 7) // 8 * 8
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
