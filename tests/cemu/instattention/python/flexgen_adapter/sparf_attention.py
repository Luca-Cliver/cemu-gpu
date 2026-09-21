from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from cemu_flexgen import SparfKvCacheStore
from cemu_flexgen.sparf_write_buffer import SparfWriteBuffer
from phase_profiler import profile_scope


@dataclass(frozen=True)
class SparfDecodeRequest:
    """Handle for one serialized SparF workflow on an isolated device."""
    layer: int
    future: Future

    def result(self, timeout=None):
        return self.future.result(timeout=timeout)

    def done(self):
        return self.future.done()


class SparfAttentionBackend:
    supports_pipelined_decode = False

    def __init__(self, layout, token_k_path, channel_k_path, token_v_path,
                 attention_device, replace_existing=False, logger=None, profiler=None,
                 buffer_tokens=0, buffer_path=None, device_writeback=None):
        self.layout = layout
        self.paths = tuple(Path(path) for path in (token_k_path, channel_k_path, token_v_path))
        self.attention_device = attention_device
        self.supports_async_decode = callable(getattr(attention_device, "submit_decode", None))
        self.logger = logger
        self.profiler = profiler
        self.store = SparfKvCacheStore(
            layout, *self.paths, replace_existing=replace_existing, profiler=profiler
        )
        self.buffered = bool(buffer_tokens)
        if self.buffered:
            if buffer_path is None:
                raise ValueError("buffered SparF requires an FDM buffer path")
            self.store = SparfWriteBuffer(self.store, buffer_path, buffer_tokens, device_writeback)
        elif device_writeback is not None:
            raise ValueError("device writeback requires a KV buffer")
        self._latest = {}
        self._executor = None
        self._executor_lock = Lock()
        self._pending = None
        self._opened = False

    @property
    def is_open(self):
        return self._opened and self.store.is_open and self.attention_device.is_open

    def open(self):
        with self._executor_lock:
            if self._opened:
                return self
            self.store.open()
            try:
                self.attention_device.open()
            except Exception:
                self.store.close()
                raise
            self._opened = True
        return self

    def write_prefill(self, layer, keys, values):
        with profile_scope(self.profiler, "prefill.kv_to_cpu_wait_wall"):
            keys, values = self._cache(keys), self._cache(values)
        with profile_scope(self.profiler, "prefill.nvm_write_wall", keys.nbytes + values.nbytes):
            self.store.write_tokens(layer, 0, keys, values)
        return keys.shape[0]

    def append_decode(self, layer, token, key, value):
        with profile_scope(self.profiler, "decode.kv_to_cpu_wait_wall"):
            key, value = self._cache(key), self._cache(value)
        with profile_scope(self.profiler, "decode.nvm_append_wall", key.nbytes + value.nbytes):
            self.store.write_token(layer, token, key[0], value[0])
        # Publish the current token only after the backing store accepted it;
        # a failed write must not leave Attention bound to data that is absent
        # from the cache.
        self._latest[layer] = (key[0].copy(), value[0].copy())

    def decode(self, layer, query, valid_tokens):
        inputs = self._decode_inputs(layer, query, valid_tokens)
        options = self._decode_options(layer, valid_tokens)
        return self.attention_device.run_decode(*inputs, **options)

    def _decode_options(self, layer, valid_tokens):
        options = dict(cache_paths=self.paths)
        if self.buffered:
            options["buffered_kv"] = self.store.snapshot(layer, valid_tokens)
        return options

    def _decode_inputs(self, layer, query, valid_tokens):
        # The CSD reads physical extents, bypassing the Guest page cache.
        # Publish before either synchronous execution or async submission.
        if not self.buffered:
            with profile_scope(self.profiler, "decode.kv_publish_wall"):
                self.store.publish()
        current_key, current_value = self._latest[layer]
        with profile_scope(self.profiler, "decode.query_to_cpu_wait_wall"):
            query = self._query(query)
        with profile_scope(self.profiler, "decode.value_mean_wall"):
            mean = self.store.value_mean(layer)
        return (
            query, layer, valid_tokens, mean,
            current_key, current_value,
        )

    def submit_decode(self, layer, query, valid_tokens):
        """Submit query-dependent work, keeping one request per backend.

        A shared scheduler interleaves independent microbatches on its slots;
        a legacy device runs the full workflow on one background worker.
        """
        with self._executor_lock:
            if not self.is_open:
                raise RuntimeError("SparF backend is not open")
            if self._pending is not None:
                if not self._pending.done():
                    raise RuntimeError("SparF backend already has an in-flight request")
                self._pending.result()
            # Snapshot on the producing thread: a later append must not change
            # the query/current KV/mean consumed by this request.
            inputs = tuple(value.copy() if isinstance(value, np.ndarray) else value
                           for value in self._decode_inputs(layer, query, valid_tokens))
            options = self._decode_options(layer, valid_tokens)
            if self.supports_async_decode:
                self._pending = self.attention_device.submit_decode(*inputs, **options)
            else:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="sparf-workflow"
                    )
                self._pending = self._executor.submit(
                    self.attention_device.run_decode, *inputs, **options
                )
            return SparfDecodeRequest(layer=layer, future=self._pending)

    @staticmethod
    def wait_decode(request, timeout=None):
        if not isinstance(request, SparfDecodeRequest):
            raise TypeError("request must be a SparfDecodeRequest")
        return request.result(timeout=timeout)

    def flush(self):
        with self._executor_lock:
            if self._pending is not None:
                self._pending.result()
                self._pending = None
            self.store.flush()

    def close(self):
        with self._executor_lock:
            if not self._opened:
                return
            # Release this backend's scheduler reference exactly once, even
            # when an in-flight request or resource teardown raises.
            self._opened = False
            executor, self._executor = self._executor, None
            pending, self._pending = self._pending, None
            try:
                if pending is not None:
                    pending.result()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
                try:
                    self.attention_device.close()
                finally:
                    self.store.close()
                    self._latest.clear()

    def __enter__(self): return self.open()
    def __exit__(self, exc_type, exc_value, traceback): self.close()

    def _cache(self, value):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        config = self.layout.config
        if array.ndim == 3:
            array = array.reshape(array.shape[0], config.batch_size,
                                  config.num_kv_heads, config.head_dim)
        return np.ascontiguousarray(array, dtype=config.dtype)

    def _query(self, value):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array.reshape(self.layout.config.batch_size, -1,
                                  self.layout.config.head_dim)
        return np.ascontiguousarray(array, dtype=self.layout.config.dtype)


class SparfGroupedAttentionBackend:
    """OPT-facing adapter for a shared device-owned grouped KV session.

    Streams are stable for the complete run: ``(layer, batch-in-microbatch,
    kv-head)`` is flattened below the backend's base.  The adapter keeps only
    an incremental value mean; historical KV is never read back to the Guest.
    """

    supports_pipelined_decode = False
    supports_async_decode = True

    def __init__(self, layout, grouped_store, attention_device, stream_base,
                 logger=None, profiler=None):
        if grouped_store is None:
            raise ValueError("grouped storage is required")
        if not isinstance(stream_base, int) or stream_base < 0:
            raise ValueError("stream_base must be a nonnegative integer")
        self.layout = layout
        self.grouped_store = grouped_store
        self.attention_device = attention_device
        self.stream_base = stream_base
        self.logger = logger
        self.profiler = profiler
        self._opened = False
        self._mean_sum = {}
        self._mean_count = {}
        self._pending = None
        self._lock = Lock()
        self._failure = None
        self._closed = False

    @property
    def store(self):
        # Used by generic diagnostics; grouped storage owns the actual pool.
        return self.grouped_store

    @property
    def is_open(self):
        return self._opened and self.grouped_store.registered and self.attention_device.is_open

    def open(self):
        with self._lock:
            if self._opened:
                return self
            if self._closed:
                raise RuntimeError("create a new grouped backend and session after close")
            if not self.grouped_store.registered:
                raise RuntimeError("open grouped storage before Attention backends")
            self.attention_device.open()
            self._failure = None
            self._mean_sum.clear()
            self._mean_count.clear()
            self._opened = True
        return self

    def _stream(self, layer, batch, kv_head):
        config = self.layout.config
        if not 0 <= layer < config.num_layers or not 0 <= batch < config.batch_size or \
                not 0 <= kv_head < config.num_kv_heads:
            raise IndexError("grouped KV stream coordinate is out of range")
        return self.stream_base + (layer * config.batch_size + batch) * config.num_kv_heads + kv_head

    def _kv(self, value, name):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        config = self.layout.config
        if array.ndim == 3:
            array = array.reshape(array.shape[0], config.batch_size,
                                  config.num_kv_heads, config.head_dim)
        if array.ndim != 4 or array.shape[1:] != (
                config.batch_size, config.num_kv_heads, config.head_dim):
            raise ValueError(f"{name} must have shape [tokens, batch, kv_heads, head_dim]")
        return np.ascontiguousarray(array, dtype=config.dtype)

    def _update_mean(self, layer, values):
        values64 = values.astype(np.float64, copy=False)
        total = values64.sum(axis=0)
        count = values.shape[0]
        if layer in self._mean_sum:
            self._mean_sum[layer] += total
            self._mean_count[layer] += count
        else:
            self._mean_sum[layer] = total.copy()
            self._mean_count[layer] = count

    def write_prefill(self, layer, keys, values):
        with self._lock:
            self._require_ready()
            self._require_layer(layer)
            if self._mean_count.get(layer, 0):
                raise RuntimeError(f"prefill for layer {layer} was already accepted")
            if self._pending is not None:
                if not self._pending.done():
                    raise RuntimeError("cannot append grouped KV while Attention is in flight")
                self._wait_pending_locked()
            with profile_scope(self.profiler, "prefill.grouped_append_wall"):
                keys = self._kv(keys, "keys")
                values = self._kv(values, "values")
                if keys.shape != values.shape or not len(keys):
                    raise ValueError("Prefill K/V must contain matching nonempty token arrays")
                if len(keys) > self.layout.config.max_seq_len:
                    raise ValueError("Prefill exceeds grouped layout")
                self._append(layer, 0, keys, values)
            return keys.shape[0]

    def append_decode(self, layer, token, key, value):
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError("decode token must be a nonnegative integer")
        with self._lock:
            self._require_ready()
            self._require_layer(layer)
            if self._pending is not None:
                if not self._pending.done():
                    raise RuntimeError("cannot append grouped KV while Attention is in flight")
                self._wait_pending_locked()
            expected = self._mean_count.get(layer, 0)
            if not expected:
                raise RuntimeError(f"layer {layer} has no prefill prefix")
            if token != expected:
                raise ValueError(f"decode token {token} is not the next accepted token {expected}")
            with profile_scope(self.profiler, "decode.grouped_append_wall"):
                keys = self._kv(key, "key")
                values = self._kv(value, "value")
                if keys.shape != values.shape or keys.shape[0] != 1:
                    raise ValueError("Decode K/V must contain one matching token")
                if token >= self.layout.config.max_seq_len:
                    raise ValueError("decode token exceeds grouped layout")
                self._append(layer, token, keys, values)

    def _append(self, layer, token, keys, values):
        # A tiled append can commit a prefix before failing. Never retry it
        # using a Guest mean that still describes the previous prefix.
        try:
            self.grouped_store.append_batch(
                self._stream(layer, 0, 0), token,
                keys.reshape(len(keys), -1, keys.shape[-1]),
                values.reshape(len(values), -1, values.shape[-1]),
            )
            self._update_mean(layer, values)
        except BaseException as error:
            self._failure = error
            raise

    def _query(self, value):
        if hasattr(value, "detach"):
            value = value.detach().contiguous().cpu().numpy()
        array = np.asarray(value)
        config = self.layout.config
        if array.ndim == 3 and array.shape[1] == 1:
            array = array.reshape(config.batch_size, -1, config.head_dim)
        return np.ascontiguousarray(array, dtype=config.dtype)

    def _mean(self, layer):
        config = self.layout.config
        if layer not in self._mean_sum or not self._mean_count[layer]:
            raise RuntimeError(f"no value mean is available for layer {layer}")
        return np.ascontiguousarray(
            (self._mean_sum[layer] / self._mean_count[layer]).astype(config.dtype),
        )

    def _decode_args(self, layer, query, valid_tokens):
        self._require_layer(layer)
        if (type(valid_tokens) is not int or
                not 0 < valid_tokens <= self.layout.config.max_seq_len):
            raise ValueError("valid_tokens is outside grouped layout")
        accepted = self._mean_count.get(layer, 0)
        if valid_tokens != accepted:
            raise ValueError(
                f"valid_tokens={valid_tokens} does not match accepted KV prefix {accepted}"
            )
        return self._query(query).copy(), layer, valid_tokens, self._mean(layer).copy()

    def decode(self, layer, query, valid_tokens):
        with self._lock:
            self._require_ready()
            if self._pending is not None:
                if not self._pending.done():
                    raise RuntimeError("grouped backend already has an in-flight request")
                self._wait_pending_locked()
            args = self._decode_args(layer, query, valid_tokens)
            try:
                return self.attention_device.run_decode(
                    *args, grouped_stream_base=self.stream_base,
                )
            except BaseException as error:
                self._failure = error
                raise

    def submit_decode(self, layer, query, valid_tokens):
        with self._lock:
            self._require_ready()
            if self._pending is not None and not self._pending.done():
                raise RuntimeError("grouped backend already has an in-flight request")
            if self._pending is not None:
                self._wait_pending_locked()
            args = self._decode_args(layer, query, valid_tokens)
            try:
                self._pending = self.attention_device.submit_decode(
                    *args, grouped_stream_base=self.stream_base,
                )
            except BaseException as error:
                self._failure = error
                raise
            return SparfDecodeRequest(layer=layer, future=self._pending)

    @staticmethod
    def wait_decode(request, timeout=None):
        if not isinstance(request, SparfDecodeRequest):
            raise TypeError("request must be a SparfDecodeRequest")
        return request.result(timeout=timeout)

    def flush(self):
        with self._lock:
            self._require_ready()
            self._wait_pending_locked()
            try:
                self.grouped_store.flush()
            except BaseException as error:
                self._failure = error
                raise

    def drain(self):
        """Terminally persist all accepted groups before session shutdown."""
        with self._lock:
            self._require_ready()
            self._wait_pending_locked()
            try:
                return self.grouped_store.drain()
            except BaseException as error:
                self._failure = error
                raise

    def close(self):
        with self._lock:
            if not self._opened:
                return
            self._opened = False
            self._closed = True
            pending = self._pending
            self._pending = None
            pending_error = None
            try:
                if pending is not None:
                    try:
                        pending.result()
                    except BaseException as error:
                        pending_error = error
            finally:
                try:
                    self.attention_device.close()
                finally:
                    self._mean_sum.clear()
                    self._mean_count.clear()
                    self._failure = None
            if pending_error is not None:
                raise pending_error

    def _require_layer(self, layer):
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < self.layout.config.num_layers:
            raise IndexError("grouped KV layer is out of range")

    def _require_ready(self):
        if not self.is_open:
            raise RuntimeError("grouped SparF backend is not open")
        if self._failure is not None:
            raise RuntimeError("grouped SparF backend is poisoned after a prior failure") from self._failure

    def _wait_pending_locked(self):
        pending = self._pending
        if pending is None:
            return None
        try:
            return pending.result()
        except BaseException as error:
            self._failure = error
            raise
        finally:
            self._pending = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *arguments):
        self.close()
