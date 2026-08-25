import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch

from .model_config import FlexGenLlamaConfig
from .weights import (
    FlexGenAttentionWeights,
    FlexGenLayerWeights,
    FlexGenMlpWeights,
    FlexGenWeightLoader,
)

try:
    from safetensors import safe_open as _safe_open
except ImportError:
    _safe_open = None


class FlexGenHfCheckpointLoader(FlexGenWeightLoader):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        model_directory: Any,
        device: Any = "cpu",
        cache_layers: bool = True,
    ):
        super().__init__(config, model_directory, device=device)
        if _safe_open is None:
            raise RuntimeError(
                "safetensors is required to load Hugging Face checkpoints"
            )
        if not isinstance(cache_layers, bool):
            raise TypeError("cache_layers must be a boolean")

        self.model_directory = self.weight_directory.resolve()
        self.cache_layers = cache_layers
        self._exit_stack = ExitStack()
        self._shard_handles: Dict[str, Any] = {}
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._layer_cache: Dict[int, FlexGenLayerWeights] = {}
        self._closed = False
        self._weight_map = self._discover_weight_map()

    @property
    def cached_layer_count(self) -> int:
        return len(self._layer_cache)

    def load_embedding(self) -> torch.Tensor:
        return self._load_cached_tensor(
            "model.embed_tokens.weight",
            (self.config.vocab_size, self.config.hidden_size),
        )

    def load_final_norm(self) -> torch.Tensor:
        return self._load_cached_tensor(
            "model.norm.weight",
            (self.config.hidden_size,),
        )

    def load_lm_head(self) -> torch.Tensor:
        tensor_name = "lm_head.weight"
        if tensor_name not in self._weight_map:
            return self.load_embedding()
        return self._load_cached_tensor(
            tensor_name,
            (self.config.vocab_size, self.config.hidden_size),
        )

    def load_layer(self, layer: int) -> FlexGenLayerWeights:
        self._validate_layer(layer)
        if self.cache_layers and layer in self._layer_cache:
            return self._layer_cache[layer]

        hidden_size = self.config.hidden_size
        kv_hidden_size = self.config.num_key_value_heads * self.config.head_dim
        intermediate_size = self.config.intermediate_size
        prefix = f"model.layers.{layer}."
        tensors = self._load_tensors(
            (
                (prefix + "self_attn.q_proj.weight", (hidden_size, hidden_size)),
                (prefix + "self_attn.k_proj.weight", (kv_hidden_size, hidden_size)),
                (prefix + "self_attn.v_proj.weight", (kv_hidden_size, hidden_size)),
                (prefix + "self_attn.o_proj.weight", (hidden_size, hidden_size)),
                (prefix + "input_layernorm.weight", (hidden_size,)),
                (prefix + "post_attention_layernorm.weight", (hidden_size,)),
                (prefix + "mlp.gate_proj.weight", (intermediate_size, hidden_size)),
                (prefix + "mlp.up_proj.weight", (intermediate_size, hidden_size)),
                (prefix + "mlp.down_proj.weight", (hidden_size, intermediate_size)),
            )
        )
        layer_weights = FlexGenLayerWeights(
            attention=FlexGenAttentionWeights(
                query=tensors[prefix + "self_attn.q_proj.weight"],
                key=tensors[prefix + "self_attn.k_proj.weight"],
                value=tensors[prefix + "self_attn.v_proj.weight"],
                output=tensors[prefix + "self_attn.o_proj.weight"],
                input_norm=tensors[prefix + "input_layernorm.weight"],
                post_attention_norm=tensors[
                    prefix + "post_attention_layernorm.weight"
                ],
            ),
            mlp=FlexGenMlpWeights(
                gate=tensors[prefix + "mlp.gate_proj.weight"],
                up=tensors[prefix + "mlp.up_proj.weight"],
                down=tensors[prefix + "mlp.down_proj.weight"],
            ),
        )
        if self.cache_layers:
            self._layer_cache[layer] = layer_weights
        return layer_weights

    def clear_layer_cache(self) -> None:
        self._layer_cache.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._layer_cache.clear()
        self._tensor_cache.clear()
        self._exit_stack.close()
        self._shard_handles.clear()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def _discover_weight_map(self) -> Mapping[str, str]:
        index_path = self.model_directory / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as index_file:
                index = json.load(index_file)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid safetensors index: {index_path}")
            for tensor_name, shard_name in weight_map.items():
                if not isinstance(tensor_name, str) or not tensor_name:
                    raise ValueError(f"invalid tensor name in {index_path}")
                if not isinstance(shard_name, str) or not shard_name:
                    raise ValueError(f"invalid shard name in {index_path}")
                self._resolve_shard_path(shard_name)
            return dict(weight_map)

        shard_name = "model.safetensors"
        shard_path = self.model_directory / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                "model.safetensors or model.safetensors.index.json was not found "
                f"in {self.model_directory}"
            )
        handle = self._open_shard(shard_name)
        tensor_names = tuple(handle.keys())
        if not tensor_names:
            raise ValueError(f"safetensors file contains no tensors: {shard_path}")
        return {tensor_name: shard_name for tensor_name in tensor_names}

    def _load_cached_tensor(
        self,
        tensor_name: str,
        expected_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        self._ensure_open()
        if tensor_name not in self._tensor_cache:
            self._tensor_cache[tensor_name] = self._load_tensor(
                tensor_name,
                expected_shape,
            )
        return self._tensor_cache[tensor_name]

    def _load_tensors(
        self,
        tensor_specs: Iterable[Tuple[str, Tuple[int, ...]]],
    ) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        loaded = {}
        for tensor_name, expected_shape in tensor_specs:
            loaded[tensor_name] = self._load_tensor(tensor_name, expected_shape)
        return loaded

    def _load_tensor(
        self,
        tensor_name: str,
        expected_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        try:
            shard_name = self._weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor does not exist: {tensor_name}") from error

        handle = self._open_shard(shard_name)
        tensor = handle.get_tensor(tensor_name)
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{tensor_name} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        tensor = tensor.to(device=self.device, dtype=self.config.dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def _open_shard(self, shard_name: str):
        self._ensure_open()
        if shard_name in self._shard_handles:
            return self._shard_handles[shard_name]
        shard_path = self._resolve_shard_path(shard_name)
        handle = self._exit_stack.enter_context(
            _safe_open(str(shard_path), framework="pt", device="cpu")
        )
        self._shard_handles[shard_name] = handle
        return handle

    def _resolve_shard_path(self, shard_name: str) -> Path:
        shard_path = (self.model_directory / shard_name).resolve()
        try:
            shard_path.relative_to(self.model_directory)
        except ValueError as error:
            raise ValueError(f"safetensors shard escapes model directory: {shard_name}") from error
        if shard_path.suffix != ".safetensors":
            raise ValueError(f"unsupported checkpoint shard: {shard_name}")
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {shard_path}")
        return shard_path

    def _validate_layer(self, layer: int) -> None:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if layer < 0 or layer >= self.config.num_hidden_layers:
            raise IndexError("layer is outside the model configuration")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("checkpoint loader is closed")
