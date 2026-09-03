from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Lock
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .attention_abi import DenseAttentionMetadata
from .kv_layout import KvCacheLayout


class CemuAttentionSharedWorkers:
    """Shared logical Load, Compute, and Store streams for one CSD."""

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        self.logger = logger
        self._lock = Lock()
        self._users = 0
        self._load_executor: Optional[ThreadPoolExecutor] = None
        self._compute_executor: Optional[ThreadPoolExecutor] = None
        self._store_executor: Optional[ThreadPoolExecutor] = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._users > 0

    def acquire(self) -> None:
        with self._lock:
            if self._users == 0:
                self._load_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="cemu-attention-load",
                )
                self._compute_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="cemu-attention-compute",
                )
                self._store_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="cemu-attention-store",
                )
                self._log("opened logical Load/Compute/Store streams")
            self._users += 1

    def release(self) -> None:
        with self._lock:
            if self._users == 0:
                return
            self._users -= 1
            if self._users != 0:
                return
            load_executor = self._load_executor
            compute_executor = self._compute_executor
            store_executor = self._store_executor
            self._load_executor = None
            self._compute_executor = None
            self._store_executor = None

        if load_executor is not None:
            load_executor.shutdown(wait=True)
        if compute_executor is not None:
            compute_executor.shutdown(wait=True)
        if store_executor is not None:
            store_executor.shutdown(wait=True)
        self._log("closed logical Load/Compute/Store streams")

    def submit_load(self, function, *args) -> Future:
        return self._submit("Load", self._load_executor, function, *args)

    def submit_compute(self, function, *args) -> Future:
        return self._submit("Compute", self._compute_executor, function, *args)

    def submit_store(self, function, *args) -> Future:
        return self._submit("Store", self._store_executor, function, *args)

    def _submit(self, name, executor, function, *args) -> Future:
        with self._lock:
            if self._users == 0 or executor is None:
                raise RuntimeError(f"CEMU Attention {name} stream is not open")
            return executor.submit(function, *args)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[cemu-attention-shared-workers] {message}")


@dataclass(frozen=True)
class CemuAttentionSlotRequest:
    request_id: int
    layer: int
    valid_tokens: int
    future: Future

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        return self.future.result(timeout=timeout)

    def done(self) -> bool:
        return self.future.done()


@dataclass(frozen=True)
class CemuAttentionPrefetchRequest:
    request_id: int
    layer: int
    history_tokens: int
    future: Future

    def result(self, timeout: Optional[float] = None) -> Any:
        return self.future.result(timeout=timeout)


@dataclass(frozen=True)
class _PrefetchedSlot:
    slot_index: int
    chunk: Any


class CemuAttentionSlotScheduler:
    """Schedule KV loading and CEMU execution across independent FDM slots."""

    def __init__(
        self,
        slots: Sequence[Any],
        workers: Optional[CemuAttentionSharedWorkers] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        normalized_slots = tuple(slots)
        if len(normalized_slots) < 2:
            raise ValueError("the Attention slot scheduler requires at least two FDM slots")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        first_layout = getattr(normalized_slots[0], "layout", None)
        if not isinstance(first_layout, KvCacheLayout):
            raise TypeError("each Attention slot must provide a KvCacheLayout")
        first_staging_bytes = self._staging_bytes(normalized_slots[0])
        for slot in normalized_slots[1:]:
            layout = getattr(slot, "layout", None)
            if not isinstance(layout, KvCacheLayout) or layout.config != first_layout.config:
                raise ValueError("all Attention slots must use the same KV layout")
            if self._staging_bytes(slot) != first_staging_bytes:
                raise ValueError("all Attention slots must use the same staging capacity")
        for method_name in (
            "open",
            "close",
            "stage_chunk",
            "execute_staged_chunk",
            "collect_output",
        ):
            if not all(callable(getattr(slot, method_name, None)) for slot in normalized_slots):
                raise TypeError(f"each Attention slot must provide {method_name}()")

        self.slots = normalized_slots
        self.layout = first_layout
        self.staging_bytes = first_staging_bytes
        self.logger = logger
        self.workers = workers or CemuAttentionSharedWorkers(logger=logger)
        if not isinstance(self.workers, CemuAttentionSharedWorkers):
            raise TypeError("workers must be CemuAttentionSharedWorkers")
        self._available_slots: Queue = Queue()
        self._request_lock = Lock()
        self._requests = []
        self._prefetches = []
        self._next_request_id = 0
        self._accepting = False

    @property
    def is_open(self) -> bool:
        return self._accepting

    def open(self):
        if self.is_open:
            return self

        opened_slots = []
        try:
            for slot in self.slots:
                slot.open()
                opened_slots.append(slot)
            self.workers.acquire()
        except Exception:
            for slot in reversed(opened_slots):
                slot.close()
            raise

        self._available_slots = Queue()
        for slot_index in range(len(self.slots)):
            self._available_slots.put(slot_index)
        self._accepting = True
        self._log(f"opened slots={len(self.slots)}")
        return self

    def submit_decode(
        self,
        query: Any,
        layer: int,
        valid_tokens: int,
    ) -> CemuAttentionSlotRequest:
        query_array = self._normalize_query(query)
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool):
            raise TypeError("valid_tokens must be an integer")
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if valid_tokens <= 0:
            raise ValueError("valid_tokens must be positive")

        with self._request_lock:
            if not self._accepting:
                raise RuntimeError("CEMU Attention slot scheduler is not open")
            request_id = self._next_request_id
            self._next_request_id += 1
            result_future = Future()
            request = CemuAttentionSlotRequest(
                request_id=request_id,
                layer=layer,
                valid_tokens=valid_tokens,
                future=result_future,
            )
            self._requests.append(request)
            self.workers.submit_load(
                self._load_request,
                request,
                query_array,
            )

        self._log(
            f"submit request={request_id}, layer={layer}, "
            f"valid_tokens={valid_tokens}"
        )
        return request

    def prefetch_decode(
        self,
        layer: int,
        history_tokens: int,
    ) -> CemuAttentionPrefetchRequest:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if not isinstance(history_tokens, int) or isinstance(history_tokens, bool):
            raise TypeError("history_tokens must be an integer")
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if history_tokens <= 0:
            raise ValueError("history_tokens must be positive")

        with self._request_lock:
            if not self._accepting:
                raise RuntimeError("CEMU Attention slot scheduler is not open")
            request_id = self._next_request_id
            self._next_request_id += 1
            request = CemuAttentionPrefetchRequest(
                request_id=request_id,
                layer=layer,
                history_tokens=history_tokens,
                future=Future(),
            )
            self._prefetches.append(request)
            self.workers.submit_load(self._load_prefetch, request)

        self._log(
            f"prefetch-submit request={request_id}, layer={layer}, "
            f"history_tokens={history_tokens}"
        )
        return request

    def submit_prefetched_decode(
        self,
        prefetch: CemuAttentionPrefetchRequest,
        query: Any,
        key: Any,
        value: Any,
    ) -> CemuAttentionSlotRequest:
        if not isinstance(prefetch, CemuAttentionPrefetchRequest):
            raise TypeError("prefetch must be a CemuAttentionPrefetchRequest")
        query_array = self._normalize_query(query)
        key_array = self._normalize_cache_token("key", key)
        value_array = self._normalize_cache_token("value", value)
        prefetched_slot = prefetch.result()
        result = CemuAttentionSlotRequest(
            request_id=prefetch.request_id,
            layer=prefetch.layer,
            valid_tokens=prefetch.history_tokens + 1,
            future=Future(),
        )

        with self._request_lock:
            if not self._accepting:
                self._available_slots.put(prefetched_slot.slot_index)
                raise RuntimeError("CEMU Attention slot scheduler is not open")
            if prefetch in self._prefetches:
                self._prefetches.remove(prefetch)
            self._requests.append(result)
            self.workers.submit_compute(
                self._compute_prefetched_request,
                result,
                query_array,
                key_array,
                value_array,
                prefetched_slot,
            )

        self._log(
            f"compute-submit request={result.request_id}, layer={result.layer}, "
            f"valid_tokens={result.valid_tokens}"
        )
        return result

    def submit_store(self, function, *args) -> Future:
        if not self.is_open:
            raise RuntimeError("CEMU Attention slot scheduler is not open")
        return self.workers.submit_store(function, *args)

    def wait_all(self) -> tuple:
        with self._request_lock:
            requests = tuple(self._requests)
            self._requests.clear()
        return tuple(request.result() for request in requests)

    def wait_request(
        self,
        request: CemuAttentionSlotRequest,
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        if not isinstance(request, CemuAttentionSlotRequest):
            raise TypeError("request must be a CemuAttentionSlotRequest")
        try:
            return request.result(timeout=timeout)
        finally:
            with self._request_lock:
                if request in self._requests:
                    self._requests.remove(request)

    def close(self) -> None:
        with self._request_lock:
            self._accepting = False
            self._requests.clear()
            self._prefetches.clear()
        self.workers.release()
        for slot in reversed(self.slots):
            slot.close()
        self._log("closed")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            try:
                self.wait_all()
            finally:
                self.close()
        else:
            self.close()

    def _load_request(
        self,
        request: CemuAttentionSlotRequest,
        query: np.ndarray,
    ) -> None:
        slot_index = self._available_slots.get()
        handed_to_compute = False
        try:
            slot = self.slots[slot_index]
            chunks = tuple(
                self.layout.iter_chunks(
                    layer=request.layer,
                    valid_tokens=request.valid_tokens,
                    k_staging_bytes=self.staging_bytes,
                    v_staging_bytes=self.staging_bytes,
                )
            )
            if len(chunks) != 1:
                raise ValueError(
                    "the slot-scheduled path requires one KV chunk per logical request"
                )
            chunk = chunks[0]
            metadata = DenseAttentionMetadata.from_layout(
                self.layout,
                num_query_heads=query.shape[1],
                token_count=chunk.token_count,
                reset_state=True,
                finalize=True,
            )
            self._log(
                f"load-start request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
            slot.stage_chunk(chunk)
            self._log(
                f"load-complete request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
            self.workers.submit_compute(
                self._compute_request,
                request,
                query,
                slot_index,
                chunk,
                metadata,
            )
            handed_to_compute = True
        except Exception as error:
            if not request.future.done():
                request.future.set_exception(error)
        finally:
            if not handed_to_compute:
                self._available_slots.put(slot_index)

    def _load_prefetch(self, request: CemuAttentionPrefetchRequest) -> None:
        slot_index = self._available_slots.get()
        handed_to_caller = False
        try:
            chunks = tuple(
                self.layout.iter_chunks(
                    layer=request.layer,
                    valid_tokens=request.history_tokens,
                    k_staging_bytes=self.staging_bytes,
                    v_staging_bytes=self.staging_bytes,
                )
            )
            if len(chunks) != 1:
                raise ValueError(
                    "the slot-scheduled prefetch path requires one historical KV chunk"
                )
            chunk = chunks[0]
            self._log(
                f"prefetch-start request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
            self.slots[slot_index].stage_chunk(chunk)
            request.future.set_result(_PrefetchedSlot(slot_index, chunk))
            handed_to_caller = True
            self._log(
                f"prefetch-complete request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
        except Exception as error:
            if not request.future.done():
                request.future.set_exception(error)
        finally:
            if not handed_to_caller:
                self._available_slots.put(slot_index)

    def _compute_request(
        self,
        request: CemuAttentionSlotRequest,
        query: np.ndarray,
        slot_index: int,
        chunk: Any,
        metadata: DenseAttentionMetadata,
    ) -> None:
        try:
            slot = self.slots[slot_index]
            self._log(
                f"compute-start request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
            slot.execute_staged_chunk(query, chunk, metadata)
            output = slot.collect_output(metadata)
            request.future.set_result(output)
            self._log(
                f"compute-complete request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
        except Exception as error:
            if not request.future.done():
                request.future.set_exception(error)
        finally:
            self._available_slots.put(slot_index)

    def _compute_prefetched_request(
        self,
        request: CemuAttentionSlotRequest,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        prefetched_slot: _PrefetchedSlot,
    ) -> None:
        slot_index = prefetched_slot.slot_index
        try:
            slot = self.slots[slot_index]
            chunk = slot.append_staged_token(key, value)
            metadata = DenseAttentionMetadata.from_layout(
                self.layout,
                num_query_heads=query.shape[1],
                token_count=chunk.token_count,
                reset_state=True,
                finalize=True,
            )
            self._log(
                f"compute-start request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
            slot.execute_staged_chunk(query, chunk, metadata)
            output = slot.collect_output(metadata)
            request.future.set_result(output)
            self._log(
                f"compute-complete request={request.request_id}, slot={slot_index}, "
                f"layer={request.layer}, tokens={chunk.token_count}"
            )
        except Exception as error:
            if not request.future.done():
                request.future.set_exception(error)
        finally:
            self._available_slots.put(slot_index)

    def _normalize_query(self, query: Any) -> np.ndarray:
        array = np.ascontiguousarray(query, dtype=self.layout.config.dtype)
        if array.ndim != 3:
            raise ValueError("decode query must have shape [batch, heads, dim]")
        if array.shape[0] != self.layout.config.batch_size:
            raise ValueError("query batch size does not match the KV layout")
        if array.shape[2] != self.layout.config.head_dim:
            raise ValueError("query head dimension does not match the KV layout")
        if array.shape[1] % self.layout.config.num_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        return array

    def _normalize_cache_token(self, name: str, value: Any) -> np.ndarray:
        array = np.ascontiguousarray(value, dtype=self.layout.config.dtype)
        expected_shape = (
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
        return array

    @staticmethod
    def _staging_bytes(slot: Any) -> int:
        buffers = getattr(slot, "buffers", None)
        staging_bytes = getattr(buffers, "staging_bytes", None)
        if not isinstance(staging_bytes, int) or staging_bytes <= 0:
            raise TypeError("each Attention slot must provide a staging capacity")
        return staging_bytes

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[cemu-attention-slot-scheduler] {message}")
