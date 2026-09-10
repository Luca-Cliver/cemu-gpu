"""Lazy PyTorch checkpoint loading for Hugging Face OPT models."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from .config import OptConfig
from .weights import (
    OptAttentionWeights,
    OptEmbeddingWeights,
    OptLayerNormWeights,
    OptLayerWeights,
    OptMlpWeights,
)


class OptCheckpointLoader:
    def __init__(
        self,
        config: OptConfig,
        model_directory: Any,
        device: Any = "cpu",
        cache_layers: bool = False,
        max_cached_shards: int = 1,
    ):
        if not isinstance(config, OptConfig):
            raise TypeError("config must be an OptConfig")
        if not isinstance(cache_layers, bool):
            raise TypeError("cache_layers must be a boolean")
        if (
            not isinstance(max_cached_shards, int)
            or isinstance(max_cached_shards, bool)
            or max_cached_shards <= 0
        ):
            raise ValueError("max_cached_shards must be a positive integer")

        self.config = config
        self.model_directory = Path(model_directory).resolve()
        self.device = torch.device(device)
        self.cache_layers = cache_layers
        self.max_cached_shards = max_cached_shards
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"model directory does not exist: {self.model_directory}"
            )

        self._weight_map, self._single_shard = self._discover_checkpoint()
        self._shard_cache: OrderedDict[str, Mapping[str, torch.Tensor]] = (
            OrderedDict()
        )
        self._resolved_tensor_names: Dict[str, str] = {}
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._embedding_cache: Optional[OptEmbeddingWeights] = None
        self._final_norm_cache: Optional[OptLayerNormWeights] = None
        self._layer_cache: Dict[int, OptLayerWeights] = {}
        self._closed = False

    @property
    def cached_layer_count(self) -> int:
        return len(self._layer_cache)

    @property
    def cached_shard_count(self) -> int:
        return len(self._shard_cache)

    @property
    def checkpoint_files(self) -> Tuple[Path, ...]:
        if self._single_shard is not None:
            return (self._resolve_shard_path(self._single_shard),)
        return tuple(
            self._resolve_shard_path(shard_name)
            for shard_name in sorted(set(self._weight_map.values()))
        )

    def load_embedding(self) -> OptEmbeddingWeights:
        self._ensure_open()
        if self._embedding_cache is None:
            self._embedding_cache = OptEmbeddingWeights(
                token=self._load_cached_tensor(
                    "model.decoder.embed_tokens.weight",
                    (self.config.vocab_size, self.config.word_embed_proj_dim),
                ),
                position=self._load_cached_tensor(
                    "model.decoder.embed_positions.weight",
                    (
                        self.config.max_position_embeddings
                        + self.config.position_offset,
                        self.config.hidden_size,
                    ),
                ),
            )
        return self._embedding_cache

    def load_final_norm(self) -> OptLayerNormWeights:
        self._ensure_open()
        if self._final_norm_cache is None:
            self._final_norm_cache = self._load_layer_norm(
                "model.decoder.final_layer_norm"
            )
        return self._final_norm_cache

    def load_lm_head(self) -> torch.Tensor:
        self._ensure_open()
        if self.config.tie_word_embeddings:
            return self.load_embedding().token
        tensor_name = "lm_head.weight"
        if not self._has_tensor(tensor_name):
            return self.load_embedding().token
        return self._load_cached_tensor(
            tensor_name,
            (self.config.vocab_size, self.config.word_embed_proj_dim),
        )

    def load_layer(self, layer: int) -> OptLayerWeights:
        self._ensure_open()
        self._validate_layer(layer)
        if self.cache_layers and layer in self._layer_cache:
            return self._layer_cache[layer]

        hidden_size = self.config.hidden_size
        ffn_dim = self.config.ffn_dim
        prefix = f"model.decoder.layers.{layer}."
        attention_prefix = prefix + "self_attn."
        tensor_specs = (
            (attention_prefix + "q_proj.weight", (hidden_size, hidden_size)),
            (attention_prefix + "q_proj.bias", (hidden_size,)),
            (attention_prefix + "k_proj.weight", (hidden_size, hidden_size)),
            (attention_prefix + "k_proj.bias", (hidden_size,)),
            (attention_prefix + "v_proj.weight", (hidden_size, hidden_size)),
            (attention_prefix + "v_proj.bias", (hidden_size,)),
            (attention_prefix + "out_proj.weight", (hidden_size, hidden_size)),
            (attention_prefix + "out_proj.bias", (hidden_size,)),
            (prefix + "self_attn_layer_norm.weight", (hidden_size,)),
            (prefix + "self_attn_layer_norm.bias", (hidden_size,)),
            (prefix + "fc1.weight", (ffn_dim, hidden_size)),
            (prefix + "fc1.bias", (ffn_dim,)),
            (prefix + "fc2.weight", (hidden_size, ffn_dim)),
            (prefix + "fc2.bias", (hidden_size,)),
            (prefix + "final_layer_norm.weight", (hidden_size,)),
            (prefix + "final_layer_norm.bias", (hidden_size,)),
        )
        tensors = self._load_tensors(tensor_specs)
        layer_weights = OptLayerWeights(
            attention=OptAttentionWeights(
                query=tensors[attention_prefix + "q_proj.weight"],
                query_bias=tensors[attention_prefix + "q_proj.bias"],
                key=tensors[attention_prefix + "k_proj.weight"],
                key_bias=tensors[attention_prefix + "k_proj.bias"],
                value=tensors[attention_prefix + "v_proj.weight"],
                value_bias=tensors[attention_prefix + "v_proj.bias"],
                output=tensors[attention_prefix + "out_proj.weight"],
                output_bias=tensors[attention_prefix + "out_proj.bias"],
                input_norm=OptLayerNormWeights(
                    weight=tensors[prefix + "self_attn_layer_norm.weight"],
                    bias=tensors[prefix + "self_attn_layer_norm.bias"],
                ),
            ),
            mlp=OptMlpWeights(
                input=tensors[prefix + "fc1.weight"],
                input_bias=tensors[prefix + "fc1.bias"],
                output=tensors[prefix + "fc2.weight"],
                output_bias=tensors[prefix + "fc2.bias"],
                input_norm=OptLayerNormWeights(
                    weight=tensors[prefix + "final_layer_norm.weight"],
                    bias=tensors[prefix + "final_layer_norm.bias"],
                ),
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
        self._embedding_cache = None
        self._final_norm_cache = None
        self._resolved_tensor_names.clear()
        self._tensor_cache.clear()
        self._shard_cache.clear()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def _discover_checkpoint(self):
        index_path = self.model_directory / "pytorch_model.bin.index.json"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as index_file:
                index = json.load(index_file)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid PyTorch checkpoint index: {index_path}")
            normalized_map = {}
            for tensor_name, shard_name in weight_map.items():
                if not isinstance(tensor_name, str) or not tensor_name:
                    raise ValueError(f"invalid tensor name in {index_path}")
                if not isinstance(shard_name, str) or not shard_name:
                    raise ValueError(f"invalid shard name in {index_path}")
                self._resolve_shard_path(shard_name)
                normalized_map[tensor_name] = shard_name
            return normalized_map, None

        shard_name = "pytorch_model.bin"
        self._resolve_shard_path(shard_name)
        return {}, shard_name

    def _has_tensor(self, tensor_name: str) -> bool:
        try:
            self._resolve_tensor_name(tensor_name)
        except KeyError:
            return False
        return True

    def _load_layer_norm(self, prefix: str) -> OptLayerNormWeights:
        hidden_size = self.config.hidden_size
        tensors = self._load_tensors(
            (
                (prefix + ".weight", (hidden_size,)),
                (prefix + ".bias", (hidden_size,)),
            )
        )
        return OptLayerNormWeights(
            weight=tensors[prefix + ".weight"],
            bias=tensors[prefix + ".bias"],
        )

    def _load_cached_tensor(
        self,
        tensor_name: str,
        expected_shape: Tuple[int, ...],
    ) -> torch.Tensor:
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
        loaded = {}
        for tensor_name, expected_shape in tensor_specs:
            loaded[tensor_name] = self._load_tensor(tensor_name, expected_shape)
        return loaded

    def _load_tensor(
        self,
        tensor_name: str,
        expected_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        checkpoint_name = self._resolve_tensor_name(tensor_name)
        shard_name = self._tensor_shard(checkpoint_name)
        state = self._load_shard(shard_name)
        try:
            tensor = state[checkpoint_name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor does not exist: {tensor_name}") from error
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint value is not a tensor: {tensor_name}")
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{tensor_name} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        tensor = tensor.to(device=self.device, dtype=self.config.dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def _resolve_tensor_name(self, tensor_name: str) -> str:
        resolved_name = self._resolved_tensor_names.get(tensor_name)
        if resolved_name is not None:
            return resolved_name

        candidates = [tensor_name]
        if tensor_name.startswith("model."):
            candidates.append(tensor_name[len("model.") :])
        else:
            candidates.append("model." + tensor_name)

        if self._single_shard is None:
            available = self._weight_map
        else:
            available = self._load_shard(self._single_shard)
        for candidate in candidates:
            if candidate in available:
                self._resolved_tensor_names[tensor_name] = candidate
                return candidate
        raise KeyError(f"checkpoint tensor does not exist: {tensor_name}")

    def _tensor_shard(self, tensor_name: str) -> str:
        if self._single_shard is not None:
            return self._single_shard
        try:
            return self._weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor does not exist: {tensor_name}") from error

    def _load_shard(self, shard_name: str) -> Mapping[str, torch.Tensor]:
        self._ensure_open()
        if shard_name in self._shard_cache:
            state = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = state
            return state

        shard_path = self._resolve_shard_path(shard_name)
        state = self._torch_load(shard_path)
        if not isinstance(state, Mapping):
            raise TypeError(f"checkpoint shard is not a state dictionary: {shard_path}")
        nested_state = state.get("state_dict")
        if isinstance(nested_state, Mapping):
            state = nested_state
        self._shard_cache[shard_name] = state
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return state

    @staticmethod
    def _torch_load(shard_path: Path):
        load_options = {"map_location": "cpu", "weights_only": True}
        try:
            return torch.load(shard_path, mmap=True, **load_options)
        except TypeError:
            return torch.load(shard_path, **load_options)
        except RuntimeError as error:
            if "mmap" not in str(error).lower():
                raise
            return torch.load(shard_path, **load_options)

    def _resolve_shard_path(self, shard_name: str) -> Path:
        shard_path = (self.model_directory / shard_name).resolve()
        try:
            shard_path.relative_to(self.model_directory)
        except ValueError as error:
            raise ValueError(f"checkpoint shard escapes model directory: {shard_name}") from error
        if shard_path.suffix != ".bin":
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
