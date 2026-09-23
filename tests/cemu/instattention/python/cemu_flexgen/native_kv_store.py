import secrets
import struct
import time
from threading import RLock

import numpy as np

from native_kv_client import (
    LinuxTransport, inspect, coordinates, BUFFER_REGISTER, BUFFER_APPEND,
    BUFFER_STATUS, BUFFER_DRAIN, PAGE_BYTES, INFO, VERSION,
)
from .sparf_workflow_abi import pack_grouped_sparf_workflow


class NativeKvStore:
    def __init__(self, layout, batches, layers, heads, max_tokens, microbatch_size,
                 device_path="/dev/ng0n1", control_path="/dev/nvme0c3",
                 namespace_path="/dev/ng0n3", budget=64 * 1024**2, batch_pages=32,
                 transport_factory=LinuxTransport):
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

    def _call(self, opcode, fields, payload=None, retry=False):
        deadline = time.monotonic() + 120
        while True:
            status, result, data = self._transport(opcode, fields, payload)
            if not status:
                return data
            if not (retry and result == 5 and status == 0xC):
                raise RuntimeError(f"native KV opcode={opcode:#x} status={status:#x} core={result}")
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
            return dict(zip(("accepted_stream0", "token_k", "channel_k", "token_v", "memory_bytes",
                             "write_bytes", "read_bytes", "write_batches", "pending_ready_ns", "sealed", "failed"),
                            struct.unpack_from("<11Q", data)))

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
