from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from pathlib import Path
import os

import numpy as np

from phase_profiler import profile_scope
from .kv_layout import align_up


def _aligned_buffer(size):
    allocation = np.empty(size + 4095, dtype=np.uint8)
    offset = -allocation.ctypes.data % 4096
    return allocation[offset:offset + size]


class SparfWriteBuffer:
    def __init__(self, store, path, capacity_tokens, device_writeback=None):
        if type(capacity_tokens) is not int or capacity_tokens < 2:
            raise ValueError("KV buffer requires at least two tokens per layer")
        self.store = store
        self.layout = store.layout
        self.path = Path(path)
        self.capacity = capacity_tokens
        self.group_tokens = max(1, capacity_tokens // 2)
        config = self.layout.config
        self.shape = (config.batch_size, config.num_kv_heads, config.head_dim)
        self.token_bytes = int(np.prod(self.shape)) * self.layout.element_size
        self.record_bytes = align_up(2 * self.token_bytes, 512)
        self.file_bytes = align_up(config.num_layers * capacity_tokens * self.record_bytes, 4096)
        self._write_packet = _aligned_buffer(self.record_bytes)
        self._write_packet.fill(0)
        self.profiler = store.profiler
        self._condition = Condition()
        self._accepted = [0] * config.num_layers
        self._published = [0] * config.num_layers
        self._scheduled = [0] * config.num_layers
        self._sums = np.zeros((config.num_layers,) + self.shape, dtype=np.float64)
        self._error = None
        self._descriptor = -1
        self._worker = None
        self.device_writeback = device_writeback

    @property
    def is_open(self):
        return self._descriptor >= 0 and self.store.is_open

    def open(self):
        if self.is_open:
            return self
        self.store.open()
        try:
            self._descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.posix_fallocate(self._descriptor, 0, self.file_bytes)
            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sparf-kv-writeback")
        except Exception:
            if self._descriptor >= 0:
                os.close(self._descriptor)
                self._descriptor = -1
            self.store.close()
            raise
        return self

    def _check(self):
        if not self.is_open:
            raise RuntimeError("KV buffer is not open")
        if self._error is not None:
            raise RuntimeError("KV background writeback failed") from self._error

    def write_tokens(self, layer, start_token, keys, values):
        if start_token != 0 or self._accepted[layer]:
            raise ValueError("buffered bulk write is only supported for initial Prefill")
        self.store.write_tokens(layer, start_token, keys, values)
        self._accepted[layer] = len(keys)
        self._sums[layer] = np.asarray(values, dtype=np.float64).sum(axis=0)

    def _offset(self, layer, token):
        return (layer * self.capacity + token % self.capacity) * self.record_bytes

    def write_token(self, layer, token, key, value):
        self.layout.token_head_offset(layer, 0, 0, token)
        arrays = [np.ascontiguousarray(array) for array in (key, value)]
        if any(array.shape != self.shape or array.dtype != self.layout.config.dtype for array in arrays):
            raise ValueError("buffered KV shape/dtype mismatch")
        with self._condition:
            self._check()
            if token != self._accepted[layer]:
                raise ValueError("buffered KV append must be contiguous")
            if self._published[layer] == 0:
                raise RuntimeError("publish Prefill before buffered Decode")
            if self._accepted[layer] - self._published[layer] >= self.capacity:
                with profile_scope(self.profiler, "decode.kv_buffer_backpressure_wall"):
                    while self._accepted[layer] - self._published[layer] >= self.capacity:
                        self._schedule(layer, force=True)
                        self._condition.wait()
                        self._check()
            packet = self._write_packet
            packet[:self.token_bytes] = arrays[0].view(np.uint8).reshape(-1)
            packet[self.token_bytes:2 * self.token_bytes] = arrays[1].view(np.uint8).reshape(-1)
            with profile_scope(self.profiler, "decode.kv_buffer_accept_wall", 2 * self.token_bytes):
                self._transfer(os.pwritev, packet, self._offset(layer, token))
            self._sums[layer] += arrays[1]
            self._accepted[layer] += 1
            self._schedule(layer)

    def _transfer(self, operation, buffer, offset):
        with memoryview(buffer) as view:
            done = 0
            while done < len(view):
                transferred = operation(self._descriptor, [view[done:]], offset + done)
                if transferred <= 0 or transferred % 512:
                    raise OSError("FDM buffer I/O returned an incomplete or unaligned transfer")
                done += transferred

    def _read(self, layer, start, count):
        keys = np.empty((count,) + self.shape, dtype=self.layout.config.dtype)
        values = np.empty_like(keys)
        data = _aligned_buffer(count * self.record_bytes) if count else None
        done = 0
        while done < count:
            run = min(count - done, self.capacity - (start + done) % self.capacity)
            self._transfer(os.preadv, data[:run * self.record_bytes],
                           self._offset(layer, start + done))
            pair = np.ndarray((run, 2, int(np.prod(self.shape))), dtype=keys.dtype,
                buffer=data, strides=(self.record_bytes, self.token_bytes, self.layout.element_size))
            pair = pair.reshape((run, 2) + self.shape)
            keys[done:done + run], values[done:done + run] = pair[:, 0], pair[:, 1]
            done += run
        return keys, values

    def _schedule(self, layer, force=False):
        count = self._accepted[layer] - self._scheduled[layer]
        if not count or (not force and count < self.group_tokens):
            return
        start = self._scheduled[layer]
        self._scheduled[layer] += count
        try:
            self._worker.submit(self._writeback, layer, start, count)
        except Exception as error:
            self._error = error
            self._condition.notify_all()
            raise

    def _writeback(self, layer, start, count):
        try:
            with self._condition:
                self._check()
            with profile_scope(self.profiler, "kv_writeback.total_wall", 2 * count * self.token_bytes):
                if self.device_writeback is not None:
                    self.device_writeback.writeback(self, layer, start, count)
                else:
                    keys, values = self._read(layer, start, count)
                    self.store.write_tokens(layer, start, keys, values)
                    self.store.publish()
            with self._condition:
                self._published[layer] = start + count
                self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def snapshot(self, layer, valid_tokens):
        with self._condition:
            self._check()
            if valid_tokens != self._accepted[layer]:
                raise ValueError("Attention requires the latest accepted KV prefix")
            start = self._published[layer]
            with profile_scope(self.profiler, "decode.kv_buffer_snapshot_wall"):
                keys, values = self._read(layer, start, valid_tokens - start)
            return keys, values

    def value_mean(self, layer):
        with self._condition:
            self._check()
            return (self._sums[layer] / self._accepted[layer]).astype(self.layout.config.dtype)

    def flush(self):
        with self._condition:
            self._check()
            initial = not any(self._published)
            if not initial:
                for layer in range(self.layout.config.num_layers):
                    self._schedule(layer, force=True)
                with profile_scope(self.profiler, "kv_writeback.drain_wall"):
                    while self._published != self._accepted:
                        self._condition.wait()
                        self._check()
        self.store.publish()
        if initial:
            if self.device_writeback is not None:
                try:
                    self.device_writeback.acquire(self)
                except Exception as error:
                    self._error = error
                    raise
            self._published[:] = self._accepted
            self._scheduled[:] = self._accepted

    def close(self):
        try:
            if self.is_open:
                self.flush()
        finally:
            try:
                if self._worker is not None:
                    self._worker.shutdown(wait=True)
                    self._worker = None
            finally:
                try:
                    if self.device_writeback is not None:
                        self.device_writeback.close()
                finally:
                    if self._descriptor >= 0:
                        os.close(self._descriptor)
                        self._descriptor = -1
                    self.store.close()

    def read_tokens(self, *args):
        self.flush()
        return self.store.read_tokens(*args)
