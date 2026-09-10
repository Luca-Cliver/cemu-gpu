"""Model-independent layer-weight prefetching."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ModelWeightRequest:
    layer: int
    future: Future

    def result(self) -> Any:
        return self.future.result()


class ModelWeightPrefetcher:
    def __init__(
        self,
        weight_loader: Any,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not callable(getattr(weight_loader, "load_layer", None)):
            raise TypeError("weight_loader must provide load_layer()")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        self.weight_loader = weight_loader
        self.logger = logger
        self._executor: Optional[ThreadPoolExecutor] = None

    def open(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="model-weight-load",
            )
        return self

    def prefetch(self, layer: int) -> ModelWeightRequest:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if layer < 0 or layer >= self.weight_loader.config.num_hidden_layers:
            raise IndexError("layer is outside the model configuration")
        if self._executor is None:
            raise RuntimeError("weight prefetcher is not open")
        request = ModelWeightRequest(
            layer=layer,
            future=self._executor.submit(self._load_layer, layer),
        )
        self._log(f"submit layer={layer}")
        return request

    def wait(self, request: ModelWeightRequest) -> Any:
        if not isinstance(request, ModelWeightRequest):
            raise TypeError("request must be a ModelWeightRequest")
        weights = request.result()
        self._log(f"complete layer={request.layer}")
        return weights

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _load_layer(self, layer: int) -> Any:
        self._log(f"load-start layer={layer}")
        weights = self.weight_loader.load_layer(layer)
        self._log(f"load-complete layer={layer}")
        return weights

    def __enter__(self):
        return self.open()

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[weight-prefetch] {message}")
