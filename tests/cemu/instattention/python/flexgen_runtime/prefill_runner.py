from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .embedding import run_flexgen_embedding
from .mlp import run_flexgen_mlp
from .model_config import FlexGenLlamaConfig
from .output import run_flexgen_output_head
from .prefill import run_flexgen_prefill
from .weights import FlexGenWeightLoader


@dataclass(frozen=True)
class FlexGenFullPrefillResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    next_token_ids: torch.Tensor
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]


class FlexGenPrefillRunner:
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        kv_writer: Optional[Any] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        if not isinstance(weight_loader, FlexGenWeightLoader):
            raise TypeError("weight_loader must be a FlexGenWeightLoader")
        if weight_loader.config != config:
            raise ValueError("weight loader configuration does not match the runner")
        if kv_writer is not None and not hasattr(kv_writer, "write_prefill"):
            raise TypeError("kv_writer must provide write_prefill(layer, keys, values)")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.config = config
        self.weight_loader = weight_loader
        self.kv_writer = kv_writer
        self.logger = logger

    def run(
        self,
        token_ids: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        collect_kv_cache: bool = False,
    ) -> FlexGenFullPrefillResult:
        token_ids = token_ids.to(self.weight_loader.device)
        self._log(
            f"start device={self.weight_loader.device}, "
            f"token_ids={tuple(token_ids.shape)}"
        )
        embedding_weight = self.weight_loader.load_embedding()
        hidden_states, attention_mask, position_ids = run_flexgen_embedding(
            token_ids,
            embedding_weight,
            self.config.pad_token_id,
        )
        self._log(
            f"embedding hidden={tuple(hidden_states.shape)}, "
            f"valid_tokens={int(attention_mask.sum().item())}"
        )
        del embedding_weight

        collected_cache = [] if collect_kv_cache else None
        for layer in range(self.config.num_hidden_layers):
            self._log(f"layer={layer} load weights")
            weights = self.weight_loader.load_layer(layer)
            attention = weights.attention
            attention_result = run_flexgen_prefill(
                inputs=hidden_states,
                attention_mask=attention_mask,
                query_weight=attention.query,
                key_weight=attention.key,
                value_weight=attention.value,
                output_weight=attention.output,
                input_norm_weight=attention.input_norm,
                post_attention_norm_weight=attention.post_attention_norm,
                num_heads=self.config.num_attention_heads,
                position_ids=position_ids,
                rope_theta=self.config.rope_theta,
                epsilon=self.config.rms_norm_epsilon,
            )
            self._log(
                f"layer={layer} attention hidden="
                f"{tuple(attention_result.hidden_states.shape)}, "
                f"K/V={tuple(attention_result.keys.shape)}"
            )
            if self.kv_writer is not None:
                self.kv_writer.write_prefill(
                    layer,
                    attention_result.keys,
                    attention_result.values,
                )
                self._log(f"layer={layer} KV persisted")
            if collected_cache is not None:
                collected_cache.append(
                    (attention_result.keys, attention_result.values)
                )
            hidden_states = run_flexgen_mlp(
                attention_result.hidden_states,
                weights.mlp,
            )
            self._log(f"layer={layer} MLP hidden={tuple(hidden_states.shape)}")
            del attention, attention_result, weights

        output = run_flexgen_output_head(
            hidden_states=hidden_states,
            final_norm_weight=self.weight_loader.load_final_norm(),
            lm_head_weight=self.weight_loader.load_lm_head(),
            epsilon=self.config.rms_norm_epsilon,
            do_sample=do_sample,
            temperature=temperature,
        )
        self._log(
            f"output logits={tuple(output.logits.shape)}, "
            f"next_tokens={output.next_token_ids.detach().cpu().reshape(-1).tolist()}"
        )
        return FlexGenFullPrefillResult(
            hidden_states=hidden_states,
            logits=output.logits,
            next_token_ids=output.next_token_ids,
            kv_cache=tuple(collected_cache) if collected_cache is not None else None,
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[prefill] {message}")
