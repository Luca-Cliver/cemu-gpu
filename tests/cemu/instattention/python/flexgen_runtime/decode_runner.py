from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .decode import (
    FlexGenDecodeAttentionOutput,
    finish_flexgen_decode_attention,
    prepare_flexgen_decode_attention,
)
from .embedding import run_flexgen_embedding
from .mlp import run_flexgen_mlp
from .model_config import FlexGenLlamaConfig
from .output import run_flexgen_output_head
from .weights import FlexGenWeightLoader
from .weight_prefetch import FlexGenWeightPrefetcher


@dataclass(frozen=True)
class FlexGenDecodeResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    next_token_ids: torch.Tensor
    layer_outputs: Optional[Tuple[FlexGenDecodeAttentionOutput, ...]]


class FlexGenDecodeRunner:
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        attention_backend: Any,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        if not isinstance(weight_loader, FlexGenWeightLoader):
            raise TypeError("weight_loader must be a FlexGenWeightLoader")
        if weight_loader.config != config:
            raise ValueError("weight loader configuration does not match the runner")
        for method_name in ("append_decode", "decode"):
            if not hasattr(attention_backend, method_name):
                raise TypeError(
                    f"attention_backend must provide {method_name}()"
                )
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.config = config
        self.weight_loader = weight_loader
        self.attention_backend = attention_backend
        self.logger = logger

    def run(
        self,
        token_ids: torch.Tensor,
        token_position: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        collect_layer_outputs: bool = False,
    ) -> FlexGenDecodeResult:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("Decode token_ids must have shape [batch, 1]")
        if not isinstance(token_position, int) or isinstance(token_position, bool):
            raise TypeError("token_position must be an integer")
        if token_position < 0:
            raise ValueError("token_position must be non-negative")

        token_ids = token_ids.to(self.weight_loader.device)
        start_message = (
            f"start position={token_position}, device={self.weight_loader.device}"
        )
        if not self._low_overhead_logging:
            start_message += (
                ", token_ids="
                f"{token_ids.detach().cpu().reshape(-1).tolist()}"
            )
        self._log(start_message)
        embedding_weight = self.weight_loader.load_embedding()
        hidden_states, _, _ = run_flexgen_embedding(
            token_ids,
            embedding_weight,
            self.config.pad_token_id,
        )
        self._log(f"embedding hidden={tuple(hidden_states.shape)}")
        del embedding_weight

        collected_outputs = [] if collect_layer_outputs else None
        valid_tokens = token_position + 1
        pipelined_decode = bool(
            getattr(self.attention_backend, "supports_pipelined_decode", False)
        )
        prefetch = (
            self.attention_backend.prefetch_decode(0, token_position)
            if pipelined_decode
            else None
        )
        if pipelined_decode:
            self._log(
                f"pipeline prefetch layer=0, history_tokens={token_position}"
            )
        with FlexGenWeightPrefetcher(
            self.weight_loader,
            logger=self.logger,
        ) as weight_prefetcher:
            weight_request = weight_prefetcher.prefetch(0)
            for layer in range(self.config.num_hidden_layers):
                self._log(f"layer={layer} wait prefetched weights")
                weights = weight_prefetcher.wait(weight_request)
                next_weight_request = (
                    weight_prefetcher.prefetch(layer + 1)
                    if layer + 1 < self.config.num_hidden_layers
                    else None
                )
                next_prefetch = (
                    self.attention_backend.prefetch_decode(
                        layer + 1,
                        token_position,
                    )
                    if pipelined_decode
                    and layer + 1 < self.config.num_hidden_layers
                    else None
                )
                if next_prefetch is not None:
                    self._log(
                        f"pipeline Load(next) layer={layer + 1}, before "
                        f"Compute(current) layer={layer}"
                    )
                self._log(f"QKV-start position={token_position}, layer={layer}")
                projection = prepare_flexgen_decode_attention(
                    inputs=hidden_states,
                    weights=weights.attention,
                    num_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    token_position=token_position,
                    rope_theta=self.config.rope_theta,
                    epsilon=self.config.rms_norm_epsilon,
                )
                self._log(f"QKV-complete position={token_position}, layer={layer}")
                self._log(
                    f"layer={layer} projected Q={tuple(projection.query.shape)}, "
                    f"K/V={tuple(projection.key.shape)}"
                )
                if pipelined_decode:
                    attention_request = self.attention_backend.submit_prefetched_decode(
                        prefetch=prefetch,
                        layer=layer,
                        token=token_position,
                        query=projection.query,
                        key=projection.key,
                        value=projection.value,
                        valid_tokens=valid_tokens,
                    )
                    attention_output = self.attention_backend.wait_decode(
                        attention_request
                    )
                    prefetch = next_prefetch
                else:
                    self.attention_backend.append_decode(
                        layer,
                        token_position,
                        projection.key,
                        projection.value,
                    )
                    self._log(f"layer={layer} KV appended at token={token_position}")
                    attention_output = self.attention_backend.decode(
                        layer=layer,
                        query=projection.query,
                        valid_tokens=valid_tokens,
                    )
                attention_result = finish_flexgen_decode_attention(
                    inputs=hidden_states,
                    projection=projection,
                    attention_output=attention_output,
                    weights=weights.attention,
                    epsilon=self.config.rms_norm_epsilon,
                )
                self._log(
                    f"layer={layer} attention="
                    f"{tuple(attention_result.attention_output.shape)}, "
                    f"post-Wo={tuple(attention_result.hidden_states.shape)}"
                )
                if collected_outputs is not None:
                    collected_outputs.append(attention_result)
                self._log(f"MLP-start position={token_position}, layer={layer}")
                hidden_states = run_flexgen_mlp(
                    attention_result.mlp_inputs,
                    weights.mlp,
                    residual=attention_result.hidden_states,
                )
                self._log(
                    f"MLP-complete position={token_position}, layer={layer}, "
                    f"hidden={tuple(hidden_states.shape)}"
                )
                del attention_result, projection, weights
                weight_request = next_weight_request

        output = run_flexgen_output_head(
            hidden_states=hidden_states,
            final_norm_weight=self.weight_loader.load_final_norm(),
            lm_head_weight=self.weight_loader.load_lm_head(),
            epsilon=self.config.rms_norm_epsilon,
            do_sample=do_sample,
            temperature=temperature,
        )
        output_message = f"output logits={tuple(output.logits.shape)}"
        if not self._low_overhead_logging:
            output_message += (
                ", next_tokens="
                f"{output.next_token_ids.detach().cpu().reshape(-1).tolist()}"
            )
        self._log(output_message)
        return FlexGenDecodeResult(
            hidden_states=hidden_states,
            logits=output.logits,
            next_token_ids=output.next_token_ids,
            layer_outputs=(
                tuple(collected_outputs) if collected_outputs is not None else None
            ),
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[decode] {message}")

    @property
    def _low_overhead_logging(self) -> bool:
        return bool(getattr(self.logger, "low_overhead", False))
