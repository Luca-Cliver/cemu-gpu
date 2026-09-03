from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Any, Callable, Optional

import numpy as np

from cemu_flexgen.kv_layout import KvCacheLayout
from cemu_flexgen.kv_store import KvCacheStore


@dataclass(frozen=True)
class FlexGenAttentionRequest:
    request_id: int
    decode_step: int
    layer: int
    microbatch: int
    future: Future

    def result(self, timeout: Optional[float] = None) -> np.ndarray:
        return self.future.result(timeout=timeout)

    def done(self) -> bool:
        return self.future.done()


class FlexGenAttentionBackend:
    """Adapt FlexGen KV tensors to CEMU storage and decode Attention."""

    def __init__(
        self,
        layout: KvCacheLayout,
        k_cache_path: Any = "/mnt/nvme0/k_cache",
        v_cache_path: Any = "/mnt/nvme0/v_cache",
        attention_device: Optional[Any] = None,
        attention_scheduler: Optional[Any] = None,
        replace_existing: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        if attention_device is not None and attention_scheduler is not None:
            raise ValueError(
                "attention_device and attention_scheduler are mutually exclusive"
            )
        if attention_device is not None and not callable(
            getattr(attention_device, "run_decode", None)
        ):
            raise TypeError("attention_device must provide run_decode()")
        if attention_scheduler is not None and not callable(
            getattr(attention_scheduler, "submit_decode", None)
        ):
            raise TypeError("attention_scheduler must provide submit_decode()")

        self.layout = layout
        self.k_cache_path = Path(k_cache_path)
        self.v_cache_path = Path(v_cache_path)
        self.attention_device = attention_device
        self.attention_scheduler = attention_scheduler
        self.logger = logger
        self.store = KvCacheStore(
            layout=layout,
            k_path=self.k_cache_path,
            v_path=self.v_cache_path,
            replace_existing=replace_existing,
        )
        self._executor: Optional[ThreadPoolExecutor] = None
        self._store_executor: Optional[ThreadPoolExecutor] = None
        self._store_futures = {}
        self._request_lock = Lock()
        self._requests = []
        self._next_request_id = 0

    @property
    def is_open(self) -> bool:
        return self.store.is_open

    def open(self):
        if self.is_open:
            return self
        self.store.open()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cemu-attention",
        )
        if not callable(getattr(self.attention_scheduler, "submit_store", None)):
            self._store_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cemu-attention-store",
            )
        self._log(
            f"opened K={self.k_cache_path}, V={self.v_cache_path}, "
            f"file_size={self.layout.file_size}"
        )
        return self

    def submit_attend(
        self,
        decode_step: int,
        layer: int,
        microbatch: int,
        token: int,
        query: Any,
        key: Any,
        value: Any,
        valid_tokens: int,
    ) -> FlexGenAttentionRequest:
        """Submit one logical Attention request to the single CSD worker."""
        self._require_open()
        self._require_attention_engine()
        for name, number in (
            ("decode_step", decode_step),
            ("layer", layer),
            ("microbatch", microbatch),
            ("token", token),
            ("valid_tokens", valid_tokens),
        ):
            if not isinstance(number, int) or isinstance(number, bool):
                raise TypeError(f"{name} must be an integer")
            if number < 0:
                raise ValueError(f"{name} must be non-negative")
        if valid_tokens == 0:
            raise ValueError("valid_tokens must be positive")

        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            executor = self._require_executor()
            future = executor.submit(
                self._run_attend,
                request_id,
                decode_step,
                layer,
                microbatch,
                token,
                query,
                key,
                value,
                valid_tokens,
            )
            request = FlexGenAttentionRequest(
                request_id=request_id,
                decode_step=decode_step,
                layer=layer,
                microbatch=microbatch,
                future=future,
            )
            self._requests.append(request)

        self._log(
            f"attend submit request={request_id}, step={decode_step}, "
            f"layer={layer}, microbatch={microbatch}, valid_tokens={valid_tokens}"
        )
        return request

    def wait_all(self) -> tuple:
        with self._request_lock:
            requests = tuple(self._requests)
            self._requests.clear()
        return tuple(request.result() for request in requests)

    def write_prefill(self, layer: int, keys: Any, values: Any) -> int:
        """Write FlexGen Prefill caches shaped [seq, batch * heads, dim]."""
        self._require_open()
        key_array = self._normalize_cache_tokens("keys", keys)
        value_array = self._normalize_cache_tokens("values", values)
        if key_array.shape[0] != value_array.shape[0]:
            raise ValueError("keys and values must contain the same number of tokens")

        self.store.write_tokens(layer, 0, key_array, value_array)
        self._log(
            f"write layer={layer}, tokens={key_array.shape[0]}, "
            f"shape={key_array.shape}, bytes={key_array.nbytes + value_array.nbytes}"
        )
        return key_array.shape[0]

    def append_decode(
        self,
        layer: int,
        token: int,
        key: Any,
        value: Any,
    ) -> None:
        """Append one FlexGen Decode KV pair to its logical token position."""
        self._require_open()
        key_array = self._normalize_cache_tokens("key", key)
        value_array = self._normalize_cache_tokens("value", value)
        if key_array.shape[0] != 1 or value_array.shape[0] != 1:
            raise ValueError("Decode key and value must each contain exactly one token")

        self.store.write_token(layer, token, key_array[0], value_array[0])
        self._log(f"append layer={layer}, token={token}")

    def decode(self, layer: int, query: Any, valid_tokens: int) -> np.ndarray:
        """Run CEMU Decode Attention with an unscaled FlexGen query tensor."""
        self._require_attention_engine()
        query_array = self._normalize_query(query)
        self._log(
            f"decode layer={layer}, valid_tokens={valid_tokens}, "
            f"query={query_array.shape}"
        )
        if self.attention_scheduler is not None:
            request = self.attention_scheduler.submit_decode(
                query_array,
                layer=layer,
                valid_tokens=valid_tokens,
            )
            wait_request = getattr(self.attention_scheduler, "wait_request", None)
            output = (
                wait_request(request)
                if callable(wait_request)
                else request.result()
            )
            self._log(
                f"decode complete request={request.request_id}, layer={layer}, "
                f"output={output.shape}"
            )
            return output
        return self.attention_device.run_decode(
            query_array,
            layer=layer,
            valid_tokens=valid_tokens,
        )

    @property
    def supports_pipelined_decode(self) -> bool:
        scheduler = self.attention_scheduler
        return scheduler is not None and all(
            callable(getattr(scheduler, method_name, None))
            for method_name in (
                "prefetch_decode",
                "submit_prefetched_decode",
                "wait_request",
            )
        )

    def prefetch_decode(self, layer: int, history_tokens: int):
        self._require_open()
        if not self.supports_pipelined_decode:
            raise RuntimeError("the configured Attention engine has no prefetch pipeline")
        self._wait_store(layer)
        return self.attention_scheduler.prefetch_decode(layer, history_tokens)

    def submit_prefetched_decode(
        self,
        prefetch,
        layer: int,
        token: int,
        query: Any,
        key: Any,
        value: Any,
        valid_tokens: int,
    ):
        self._require_open()
        if not self.supports_pipelined_decode:
            raise RuntimeError("the configured Attention engine has no prefetch pipeline")
        if prefetch.layer != layer:
            raise ValueError("prefetched layer does not match the Decode layer")
        if token != prefetch.history_tokens or valid_tokens != token + 1:
            raise ValueError("prefetched history does not match the Decode token position")

        query_array = self._normalize_query(query)
        key_array = self._normalize_cache_tokens("key", key)
        value_array = self._normalize_cache_tokens("value", value)
        if key_array.shape[0] != 1 or value_array.shape[0] != 1:
            raise ValueError("Decode key and value must each contain exactly one token")
        request = self.attention_scheduler.submit_prefetched_decode(
            prefetch,
            query_array,
            key_array[0],
            value_array[0],
        )

        self._log(
            f"store-submit request={request.request_id}, layer={layer}, token={token}"
        )
        with self._request_lock:
            self._store_futures[layer] = self._submit_store(
                self._store_decode_token,
                request.request_id,
                layer,
                token,
                key_array[0],
                value_array[0],
            )
        self._log(
            f"pipeline submit request={request.request_id}, layer={layer}, "
            f"token={token}, valid_tokens={valid_tokens}"
        )
        return request

    def wait_decode(self, request) -> np.ndarray:
        if not self.supports_pipelined_decode:
            raise RuntimeError("the configured Attention engine has no prefetch pipeline")
        return self.attention_scheduler.wait_request(request)

    def flush(self) -> None:
        self.wait_all()
        self._wait_all_stores()
        self.store.flush()
        self._log("flushed KV files")

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        store_executor = self._store_executor
        self._store_executor = None
        if store_executor is not None:
            store_executor.shutdown(wait=True)
        with self._request_lock:
            self._requests.clear()
            self._store_futures.clear()
        if self.store.is_open:
            self.store.close()
        self._log("closed KV files")

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

    def _run_attend(
        self,
        request_id: int,
        decode_step: int,
        layer: int,
        microbatch: int,
        token: int,
        query: Any,
        key: Any,
        value: Any,
        valid_tokens: int,
    ) -> np.ndarray:
        start_ns = perf_counter_ns()
        self._log(
            f"attend start request={request_id}, step={decode_step}, "
            f"layer={layer}, microbatch={microbatch}"
        )
        self.append_decode(layer, token, key, value)
        output = self.decode(layer, query, valid_tokens)
        elapsed_ns = perf_counter_ns() - start_ns
        self._log(
            f"attend complete request={request_id}, step={decode_step}, "
            f"layer={layer}, microbatch={microbatch}, elapsed_ns={elapsed_ns}"
        )
        return output

    def _normalize_cache_tokens(self, name: str, value: Any) -> np.ndarray:
        array = self._to_numpy(name, value)
        config = self.layout.config
        flexgen_shape = (
            None,
            config.batch_size * config.num_kv_heads,
            config.head_dim,
        )
        cemu_shape = (
            None,
            config.batch_size,
            config.num_kv_heads,
            config.head_dim,
        )

        if (
            array.ndim == 3
            and array.shape[1] == flexgen_shape[1]
            and array.shape[2] == flexgen_shape[2]
        ):
            array = array.reshape(
                array.shape[0],
                config.batch_size,
                config.num_kv_heads,
                config.head_dim,
            )
        elif array.ndim != 4 or array.shape[1:] != cemu_shape[1:]:
            raise ValueError(
                f"{name} shape {array.shape} does not match FlexGen "
                f"[tokens, {flexgen_shape[1]}, {flexgen_shape[2]}] or CEMU "
                f"[tokens, {cemu_shape[1]}, {cemu_shape[2]}, {cemu_shape[3]}]"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one token")

        return np.ascontiguousarray(array, dtype=config.dtype)

    def _normalize_query(self, value: Any) -> np.ndarray:
        array = self._to_numpy("query", value)
        batch_size = self.layout.config.batch_size
        head_dim = self.layout.config.head_dim

        if array.ndim == 3 and array.shape[0] == batch_size:
            if array.shape[2] != head_dim:
                raise ValueError("query head dimension does not match the KV layout")
        elif (
            array.ndim == 3
            and array.shape[1] == 1
            and array.shape[2] == head_dim
            and array.shape[0] % batch_size == 0
        ):
            array = array.reshape(batch_size, array.shape[0] // batch_size, head_dim)
        else:
            raise ValueError(
                "query must have shape [batch, query_heads, head_dim] or "
                "[batch * query_heads, 1, head_dim]"
            )
        return np.ascontiguousarray(array, dtype=self.layout.config.dtype)

    @staticmethod
    def _to_numpy(name: str, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray) and hasattr(value, "data"):
            value = value.data
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()

        try:
            array = np.asarray(value)
        except Exception as error:
            raise TypeError(f"{name} cannot be converted to a NumPy array") from error
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric data")
        return array

    def _require_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("FlexGen CEMU backend is not open")

    def _require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("FlexGen CEMU Attention worker is not running")
        return self._executor

    def _require_store_executor(self) -> ThreadPoolExecutor:
        if self._store_executor is None:
            raise RuntimeError("FlexGen CEMU Store worker is not running")
        return self._store_executor

    def _submit_store(self, function, *args) -> Future:
        scheduler_submit = getattr(self.attention_scheduler, "submit_store", None)
        if callable(scheduler_submit):
            return scheduler_submit(function, *args)
        return self._require_store_executor().submit(function, *args)

    def _store_decode_token(
        self,
        request_id: int,
        layer: int,
        token: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self._log(
            f"store-start request={request_id}, layer={layer}, token={token}"
        )
        self.store.write_token(layer, token, key, value)
        self._log(
            f"store-complete request={request_id}, layer={layer}, token={token}"
        )

    def _wait_store(self, layer: int) -> None:
        with self._request_lock:
            future = self._store_futures.get(layer)
        if future is None:
            return
        self._log(f"store-read-barrier layer={layer}")
        future.result()
        with self._request_lock:
            if self._store_futures.get(layer) is future:
                del self._store_futures[layer]

    def _wait_all_stores(self) -> None:
        with self._request_lock:
            futures = tuple(self._store_futures.values())
            self._store_futures.clear()
        for future in futures:
            future.result()

    def _require_attention_engine(self) -> None:
        if self.attention_device is None and self.attention_scheduler is None:
            raise RuntimeError("no CEMU attention device or scheduler is configured")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[kv-backend] {message}")
