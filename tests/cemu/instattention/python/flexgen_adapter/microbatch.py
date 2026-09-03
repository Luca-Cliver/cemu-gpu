from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class FlexGenPrefillWriteRequest:
    layer: int
    future: Future

    def result(self, timeout=None) -> int:
        return self.future.result(timeout=timeout)


class FlexGenMicrobatchKvWriter:
    def __init__(
        self,
        backends: Sequence[Any],
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("backends must not be empty")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        self.logger = logger
        first_layout = self.backends[0].layout
        self.num_kv_heads = first_layout.config.num_kv_heads
        self.gpu_batch_sizes = []
        for backend in self.backends:
            if not callable(getattr(backend, "write_prefill", None)):
                raise TypeError("each backend must provide write_prefill()")
            config = backend.layout.config
            if (
                config.num_kv_heads != self.num_kv_heads
                or config.head_dim != first_layout.config.head_dim
                or config.dtype != first_layout.config.dtype
                or config.num_layers != first_layout.config.num_layers
                or config.max_seq_len != first_layout.config.max_seq_len
            ):
                raise ValueError("microbatch backends must use compatible KV layouts")
            self.gpu_batch_sizes.append(config.batch_size)
        self.gpu_batch_sizes = tuple(self.gpu_batch_sizes)
        self._executor = None
        self._lock = Lock()
        self._requests = []

    @property
    def batch_size(self) -> int:
        return sum(self.gpu_batch_sizes)

    def write_prefill(self, layer: int, keys: Any, values: Any) -> int:
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError(
                "Prefill K/V must share shape [tokens, batch * kv_heads, head_dim]"
            )
        expected_heads = self.batch_size * self.num_kv_heads
        if keys.shape[1] != expected_heads:
            raise ValueError(
                f"Prefill K/V contain {keys.shape[1]} batch-head rows, "
                f"expected {expected_heads}"
            )

        head_start = 0
        token_count = keys.shape[0]
        self._log(
            f"store-start layer={layer}, tokens={token_count}, "
            f"microbatches={len(self.backends)}"
        )
        for backend, gpu_batch_size in zip(self.backends, self.gpu_batch_sizes):
            head_end = head_start + gpu_batch_size * self.num_kv_heads
            backend.write_prefill(
                layer,
                keys[:, head_start:head_end, :],
                values[:, head_start:head_end, :],
            )
            head_start = head_end
        self._log(
            f"store-complete layer={layer}, tokens={token_count}, "
            f"microbatches={len(self.backends)}"
        )
        return token_count

    def open(self):
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="flexgen-prefill-store",
                )
        return self

    def submit_prefill(
        self,
        layer: int,
        keys: Any,
        values: Any,
    ) -> FlexGenPrefillWriteRequest:
        self.open()
        with self._lock:
            request = FlexGenPrefillWriteRequest(
                layer=layer,
                future=self._executor.submit(
                    self.write_prefill,
                    layer,
                    keys,
                    values,
                ),
            )
            self._requests.append(request)
        return request

    def wait_prefill(self, request: FlexGenPrefillWriteRequest) -> int:
        if not isinstance(request, FlexGenPrefillWriteRequest):
            raise TypeError("request must be a FlexGenPrefillWriteRequest")
        try:
            return request.result()
        finally:
            with self._lock:
                if request in self._requests:
                    self._requests.remove(request)

    def flush(self) -> None:
        with self._lock:
            requests = tuple(self._requests)
            self._requests.clear()
        for request in requests:
            request.result()
        for backend in self.backends:
            backend.flush()

    def close(self) -> None:
        self.flush()
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[prefill-kv-writer] {message}")
