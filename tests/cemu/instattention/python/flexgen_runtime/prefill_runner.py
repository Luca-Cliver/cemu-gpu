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
        if kv_writer is not None:
            synchronous_writer = callable(getattr(kv_writer, "write_prefill", None))
            asynchronous_writer = all(
                callable(getattr(kv_writer, method_name, None))
                for method_name in ("submit_prefill", "wait_prefill")
            )
            if not synchronous_writer and not asynchronous_writer:
                raise TypeError(
                    "kv_writer must provide write_prefill() or "
                    "submit_prefill()/wait_prefill()"
                )
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
        embedding_message = f"embedding hidden={tuple(hidden_states.shape)}"
        if self.logger is not None and not self._low_overhead_logging:
            embedding_message += (
                f", valid_tokens={int(attention_mask.sum().item())}"
            )
        self._log(embedding_message)
        del embedding_weight

        collected_cache = [] if collect_kv_cache else None
        pending_write = None
        for layer in range(self.config.num_hidden_layers):
            self._log(f"layer={layer} load weights")
            weights = self.weight_loader.load_layer(layer)
            attention = weights.attention
            self._log(f"layer={layer} Attention-start")
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
                num_key_value_heads=self.config.num_key_value_heads,
                position_ids=position_ids,
                rope_theta=self.config.rope_theta,
                epsilon=self.config.rms_norm_epsilon,
            )
            self._log(
                f"layer={layer} Attention-complete hidden="
                f"{tuple(attention_result.hidden_states.shape)}, "
                f"K/V={tuple(attention_result.keys.shape)}"
            )
            if self.kv_writer is not None:
                submit_prefill = getattr(self.kv_writer, "submit_prefill", None)
                wait_prefill = getattr(self.kv_writer, "wait_prefill", None)
                if callable(submit_prefill) and callable(wait_prefill):
                    current_write = submit_prefill(
                        layer,
                        attention_result.keys,
                        attention_result.values,
                    )
                    self._log(f"layer={layer} KV Store submitted")
                    if pending_write is not None:
                        completed_layer = pending_write.layer
                        wait_prefill(pending_write)
                        self._log(
                            f"layer={completed_layer} KV Store joined after "
                            f"layer={layer} Attention"
                        )
                    pending_write = current_write
                else:
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
            self._log(f"layer={layer} MLP-start")
            hidden_states = run_flexgen_mlp(
                attention_result.mlp_inputs,
                weights.mlp,
                residual=attention_result.hidden_states,
            )
            self._log(
                f"layer={layer} MLP-complete hidden={tuple(hidden_states.shape)}"
            )
            del attention, attention_result, weights

        output = run_flexgen_output_head(
            hidden_states=hidden_states,
            final_norm_weight=self.weight_loader.load_final_norm(),
            lm_head_weight=self.weight_loader.load_lm_head(),
            epsilon=self.config.rms_norm_epsilon,
            do_sample=do_sample,
            temperature=temperature,
        )
        if pending_write is not None:
            completed_layer = pending_write.layer
            self.kv_writer.wait_prefill(pending_write)
            self._log(f"layer={completed_layer} KV Store completed")
        output_message = f"output logits={tuple(output.logits.shape)}"
        if not self._low_overhead_logging:
            output_message += (
                ", next_tokens="
                f"{output.next_token_ids.detach().cpu().reshape(-1).tolist()}"
            )
        self._log(output_message)
        return FlexGenFullPrefillResult(
            hidden_states=hidden_states,
            logits=output.logits,
            next_token_ids=output.next_token_ids,
            kv_cache=tuple(collected_cache) if collected_cache is not None else None,
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[prefill] {message}")

    @property
    def _low_overhead_logging(self) -> bool:
        return bool(getattr(self.logger, "low_overhead", False))
