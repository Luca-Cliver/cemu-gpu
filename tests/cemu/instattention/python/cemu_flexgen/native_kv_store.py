import secrets
import struct
import time
from threading import RLock

import numpy as np

from native_kv_client import (
    LinuxTransport, inspect, coordinates, BUFFER_REGISTER, BUFFER_APPEND,
    BUFFER_STATUS, BUFFER_DRAIN, PAGE_BYTES, INFO, VERSION,
    BUFFER_APPEND_BATCH, APPEND_MAX_BYTES,
)
from .sparf_workflow_abi import pack_grouped_sparf_workflow


class NativeKvStore:
    def __init__(self, layout, batches, layers, heads, max_tokens, microbatch_size,
                 device_path="/dev/ng0n1", control_path="/dev/nvme0c3",
                 namespace_path="/dev/ng0n3", budget=64 * 1024**2, batch_pages=32,
                 transport_factory=LinuxTransport, append_batch_bytes=None):
        if (min(batches, layers, heads, max_tokens, microbatch_size, budget) <= 0 or
                batches % microbatch_size or max(batches, layers, heads) > 65536 or
                layout.head_dim > 256 or not layout.channel_group_pages <= batch_pages <= 256):
            raise ValueError("invalid native KV geometry or write batch size")
        self.layout = layout
        self.batches, self.layers, self.heads = batches, layers, heads
        self.max_tokens, self.microbatch_size = max_tokens, microbatch_size
        self.buffer_streams = batches * layers * heads
        self.control_path, self.namespace_path = control_path, namespace_path
        self.device_path, self.budget, self.batch_pages = device_path, budget, batch_pages
        self.task, self.generation = secrets.randbits(32) or 1, 1
        self.registered = False
        self._transport = None
        self._factory = transport_factory
        self._lock = RLock()
        self._failure = None
        self._sealed = False
        self._accepted = {}
        self._opened = False
        self._append_attempts = 0
        self._append_commands = 0
        self._append_retries = 0
        self._appended_head_tokens = 0
        if append_batch_bytes is not None and not 0 <= append_batch_bytes <= APPEND_MAX_BYTES:
            raise ValueError("invalid native APPEND transfer limit")
        self._append_requested_bytes = append_batch_bytes
        self._append_batch_bytes = 0

    def _call(self, opcode, fields, payload=None, retry=False):
        deadline = time.monotonic() + 120
        append = opcode in (BUFFER_APPEND, BUFFER_APPEND_BATCH)
        while True:
            if append:
                self._append_attempts += 1
            status, result, data = self._transport(opcode, fields, payload)
            if not status:
                if append:
                    self._append_commands += 1
                    self._appended_head_tokens += (fields[0] >> 16) * (
                        fields[4] >> 16 if opcode == BUFFER_APPEND_BATCH else 1)
                return data
            if not (retry and result == 5 and status == 0xC):
                raise RuntimeError(f"native KV opcode={opcode:#x} status={status:#x} core={result}")
            if append:
                self._append_retries += 1
            if time.monotonic() >= deadline:
                raise TimeoutError("native KV backpressure did not clear within 120 seconds")
            time.sleep(0.0001)

    def open(self):
        with self._lock:
            if self.registered:
                return self
            if self._opened or self._failure:
                raise RuntimeError("native sessions cannot be reopened; start a fresh controller")
            self._transport = self._factory(self.device_path)
            try:
                inspect(self._transport)
                info = self._call(INFO, (VERSION, 0, 0, 0, 0, 0))
                available, page_limit, compute = struct.unpack_from("<3Q", info, 80)
                append_limit = struct.unpack_from("<Q", info, 104)[0]
                requested = self._append_requested_bytes
                self._append_batch_bytes = min(append_limit, APPEND_MAX_BYTES,
                    APPEND_MAX_BYTES if requested is None else requested)
                if 0 < self._append_batch_bytes < 2 * self.layout.head_dim * self.layout.dtype.itemsize:
                    raise ValueError("native APPEND limit cannot fit one K/V row")
                token_pages = (self.max_tokens + self.layout.token_group_tokens - 1) // self.layout.token_group_tokens
                channel_groups = (self.max_tokens + self.layout.channel_group_tokens - 1) // self.layout.channel_group_tokens
                required_pages = self.buffer_streams * (2 * token_pages + channel_groups * self.layout.channel_group_pages)
                if not compute or self.budget > available or required_pages > page_limit:
                    raise ValueError(f"native controller capacity: compute={compute}, buffer_budget={available}, "
                                     f"page_limit={page_limit}; requested budget={self.budget}, pages={required_pages}")
                packet = struct.pack("<8IQI", self.batches, self.layers, self.heads, self.layout.head_dim,
                                     self.max_tokens, self.layout.channels_per_page, self.layout.dtype.itemsize,
                                     self.layout.page_bytes, self.budget, self.batch_pages).ljust(PAGE_BYTES, b"\0")
                self._call(BUFFER_REGISTER, coordinates(self.task, self.generation), packet)
                self.registered = True
                self._opened = True
            except BaseException:
                self._transport.close()
                self._transport = None
                raise
            return self

    def check_ready(self):
        if not self.registered or self._failure:
            raise RuntimeError("native KV session is closed or failed") from self._failure

    def _coordinate(self, stream):
        per_microbatch = self.layers * self.microbatch_size * self.heads
        microbatch, local = divmod(stream, per_microbatch)
        layer, local = divmod(local, self.microbatch_size * self.heads)
        batch, head = divmod(local, self.heads)
        return microbatch * self.microbatch_size + batch, layer, head

    def append_batch(self, first_stream, token, keys, values):
        keys, values = (np.ascontiguousarray(array, dtype=self.layout.dtype) for array in (keys, values))
        if (keys.ndim != 3 or keys.shape != values.shape or keys.shape[2] != self.layout.head_dim or
                not keys.shape[0] or not keys.shape[1] or first_stream < 0 or
                first_stream + keys.shape[1] > self.buffer_streams or token < 0 or
                token + len(keys) > self.max_tokens):
            raise ValueError("native APPEND requires bounded [tokens, streams, head_dim] K/V")
        with self._lock:
            self.check_ready()
            if self._sealed or any(self._accepted.get(stream, 0) != token
                                  for stream in range(first_stream, first_stream + keys.shape[1])):
                raise ValueError("native APPEND must extend every stream's accepted prefix")
            tile = PAGE_BYTES // (2 * self.layout.head_dim * self.layout.dtype.itemsize)
            try:
                if self._append_batch_bytes:
                    self._append_heads(first_stream, token, keys, values)
                    return
                for column in range(keys.shape[1]):
                    stream = first_stream + column
                    batch, layer, head = self._coordinate(stream)
                    for start in range(0, len(keys), tile):
                        count = min(tile, len(keys) - start)
                        packet = (keys[start:start + count, column].tobytes() +
                                  values[start:start + count, column].tobytes()).ljust(PAGE_BYTES, b"\0")
                        self._call(BUFFER_APPEND, coordinates(self.task, self.generation,
                            batch=batch, layer=layer, head=head, token=token + start, valid=count), packet, retry=True)
                    self._accepted[stream] = token + len(keys)
            except BaseException as error:
                self._failure = error
                raise

    def _append_heads(self, first_stream, token, keys, values):
        row_bytes = 2 * self.layout.head_dim * self.layout.dtype.itemsize
        tile = min(self.layout.token_group_tokens, self._append_batch_bytes // row_bytes)
        for start in range(0, len(keys), tile):
            count = min(tile, len(keys) - start)
            column = 0
            while column < keys.shape[1]:
                batch, layer, head = self._coordinate(first_stream + column)
                heads = min(self.heads - head, keys.shape[1] - column,
                            self._append_batch_bytes // (count * row_bytes))
                fields = list(coordinates(self.task, self.generation, batch=batch, layer=layer,
                                          head=head, token=token + start, valid=count))
                fields[4] |= heads << 16
                packet = b"".join(array[start:start + count, column:column + heads]
                                  .transpose(1, 0, 2).tobytes() for array in (keys, values))
                self._call(BUFFER_APPEND_BATCH, tuple(fields), packet, retry=True)
                for stream in range(first_stream + column, first_stream + column + heads):
                    self._accepted[stream] = token + start + count
                column += heads

    def pack_workflow(self, layout, layer, tokens, query_heads, top_r, top_k, stream_base, **kwargs):
        self.check_ready()
        per_microbatch = self.layers * self.microbatch_size * self.heads
        if (stream_base < 0 or stream_base + per_microbatch > self.buffer_streams or
                stream_base % per_microbatch):
            raise ValueError("native Attention must start at a complete microbatch")
        metadata = bytearray(pack_grouped_sparf_workflow(
            layout, layer, tokens, query_heads, top_r, top_k, 1, 0, **kwargs))
        flags = struct.unpack_from("<I", metadata, 4)[0]
        struct.pack_into("<I", metadata, 4, flags | 8)
        metadata[-16:] = struct.pack("<4I", self.task, self.generation,
                                    stream_base // per_microbatch * self.microbatch_size, 0)
        return bytes(metadata)

    def flush(self):
        self.check_ready()

    def drain(self):
        with self._lock:
            self.check_ready()
            self._call(BUFFER_DRAIN, coordinates(self.task, self.generation), retry=True)
            self._sealed = True

    def statistics(self):
        with self._lock:
            self.check_ready()
            data = self._call(BUFFER_STATUS, coordinates(self.task, self.generation))
            stats = dict(zip(("accepted_stream0", "token_k", "channel_k", "token_v", "memory_bytes",
                             "write_bytes", "read_bytes", "write_batches", "pending_ready_ns", "sealed", "failed"),
                            struct.unpack_from("<11Q", data)))
            stats.update(client_append_attempts=self._append_attempts,
                         client_append_batch_max_bytes=self._append_batch_bytes,
                         client_append_commands=self._append_commands,
                         client_append_backpressure_retries=self._append_retries,
                         client_appended_head_tokens=self._appended_head_tokens)
            return stats

    def close(self):
        with self._lock:
            if self._transport is not None:
                self._transport.close()
                self._transport = None
            self.registered = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
