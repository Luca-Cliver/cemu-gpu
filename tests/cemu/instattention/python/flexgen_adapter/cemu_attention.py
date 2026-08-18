from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from cemu_flexgen.kv_layout import KvCacheLayout
from cemu_flexgen.kv_store import KvCacheStore


class FlexGenAttentionBackend:
    """Adapt FlexGen KV tensors to CEMU storage and decode Attention."""

    def __init__(
        self,
        layout: KvCacheLayout,
        k_cache_path: Any = "/mnt/nvme0/k_cache",
        v_cache_path: Any = "/mnt/nvme0/v_cache",
        attention_device: Optional[Any] = None,
        replace_existing: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.layout = layout
        self.k_cache_path = Path(k_cache_path)
        self.v_cache_path = Path(v_cache_path)
        self.attention_device = attention_device
        self.logger = logger
        self.store = KvCacheStore(
            layout=layout,
            k_path=self.k_cache_path,
            v_path=self.v_cache_path,
            replace_existing=replace_existing,
        )

    @property
    def is_open(self) -> bool:
        return self.store.is_open

    def open(self):
        self.store.open()
        self._log(
            f"opened K={self.k_cache_path}, V={self.v_cache_path}, "
            f"file_size={self.layout.file_size}"
        )
        return self

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
        if self.attention_device is None:
            raise RuntimeError("no CEMU attention device is configured")
        query_array = self._normalize_query(query)
        self._log(
            f"decode layer={layer}, valid_tokens={valid_tokens}, "
            f"query={query_array.shape}"
        )
        return self.attention_device.run_decode(
            query_array,
            layer=layer,
            valid_tokens=valid_tokens,
        )

    def flush(self) -> None:
        self.store.flush()
        self._log("flushed KV files")

    def close(self) -> None:
        self.store.close()
        self._log("closed KV files")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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
        return np.ascontiguousarray(array, dtype=np.float32)

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

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[kv-backend] {message}")
