"""Model-independent Prefill orchestration."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .model_ops import ModelOperations, validate_model_operations


@dataclass(frozen=True)
class ModelPrefillResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    next_token_ids: torch.Tensor
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]


class ModelPrefillRunner:
    def __init__(
        self,
        config: Any,
        weight_loader: Any,
        operations: ModelOperations,
        kv_writer: Optional[Any] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not hasattr(config, "num_hidden_layers"):
            raise TypeError("config must provide num_hidden_layers")
        for method_name in (
            "load_embedding",
            "load_layer",
            "load_final_norm",
            "load_lm_head",
        ):
            if not callable(getattr(weight_loader, method_name, None)):
                raise TypeError(f"weight_loader must provide {method_name}()")
        if weight_loader.config != config:
            raise ValueError("weight loader configuration does not match the runner")
        validate_model_operations(operations)
        if kv_writer is not None:
            synchronous_writer = callable(getattr(kv_writer, "write_prefill", None))
            asynchronous_writer = all(
                callable(getattr(kv_writer, method_name, None))
                for method_name in ("submit_prefill", "wait_prefill")
            )
            microbatch_writer = all(
                callable(getattr(kv_writer, method_name, None))
                for method_name in (
                    "submit_prefill_microbatch",
                    "wait_prefill",
                )
            )
            if not synchronous_writer and not asynchronous_writer and not microbatch_writer:
                raise TypeError(
                    "kv_writer must provide write_prefill() or "
                    "submit_prefill()/wait_prefill() or "
                    "submit_prefill_microbatch()/wait_prefill()"
                )
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.config = config
        self.weight_loader = weight_loader
        self.operations = operations
        self.kv_writer = kv_writer
        self.logger = logger

    def run(
        self,
        token_ids: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        collect_kv_cache: bool = False,
        last_token_only: bool = False,
    ) -> ModelPrefillResult:
        token_ids = token_ids.to(self.weight_loader.device)
        self._log(
            f"start device={self.weight_loader.device}, "
            f"token_ids={tuple(token_ids.shape)}"
        )
        embedding_weights = self.weight_loader.load_embedding()
        embedding = self.operations.embed(
            token_ids,
            embedding_weights,
        )
        hidden_states = embedding.hidden_states
        attention_mask = embedding.attention_mask
        position_ids = embedding.position_ids
        embedding_message = f"embedding hidden={tuple(hidden_states.shape)}"
        if self.logger is not None and not self._low_overhead_logging:
            embedding_message += (
                f", valid_tokens={int(attention_mask.sum().item())}"
            )
        self._log(embedding_message)
        del embedding_weights, embedding

        collected_cache = [] if collect_kv_cache else None
        pending_write = None
        for layer in range(self.config.num_hidden_layers):
            self._log(f"layer={layer} load weights")
            weights = self.weight_loader.load_layer(layer)
            self._log(f"layer={layer} Attention-start")
            attention_result = self.operations.prefill_attention(
                hidden_states,
                attention_mask,
                position_ids,
                weights,
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
            hidden_states = self.operations.run_mlp(
                attention_result.mlp_inputs,
                weights.mlp,
                residual=attention_result.hidden_states,
            )
            self._log(
                f"layer={layer} MLP-complete hidden={tuple(hidden_states.shape)}"
            )
            del attention_result, weights

        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        output = self.operations.run_output_head(
            hidden_states,
            self.weight_loader.load_final_norm(),
            self.weight_loader.load_lm_head(),
            do_sample,
            temperature,
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
        return ModelPrefillResult(
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
