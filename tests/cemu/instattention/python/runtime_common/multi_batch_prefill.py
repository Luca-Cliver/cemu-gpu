"""Model-independent layer-major, multi-microbatch Prefill scheduling."""

from typing import Any, Callable, Optional

import torch

from .model_ops import ModelOperations
from .prefill_runner import ModelPrefillResult, ModelPrefillRunner


class ModelMultiBatchPrefillRunner(ModelPrefillRunner):
    def __init__(
        self,
        config: Any,
        weight_loader: Any,
        operations: ModelOperations,
        kv_writer: Any,
        gpu_batch_size: int,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if (
            not isinstance(gpu_batch_size, int)
            or isinstance(gpu_batch_size, bool)
            or gpu_batch_size <= 0
        ):
            raise ValueError("gpu_batch_size must be a positive integer")
        for method_name in (
            "submit_prefill_microbatch",
            "wait_prefill",
        ):
            if not callable(getattr(kv_writer, method_name, None)):
                raise TypeError(f"kv_writer must provide {method_name}()")
        super().__init__(
            config,
            weight_loader,
            operations,
            kv_writer=kv_writer,
            logger=logger,
        )
        self.gpu_batch_size = gpu_batch_size

    def run(
        self,
        token_ids: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        collect_kv_cache: bool = False,
        last_token_only: bool = False,
    ) -> ModelPrefillResult:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[0] % self.gpu_batch_size != 0:
            raise ValueError("batch size must be divisible by gpu_batch_size")
        token_ids = token_ids.to(self.weight_loader.device)
        token_batches = token_ids.split(self.gpu_batch_size, dim=0)
        if len(token_batches) != len(self.kv_writer.backends):
            raise ValueError("KV backends do not match the number of GPU batches")

        embedding_weights = self.weight_loader.load_embedding()
        embeddings = tuple(
            self.operations.embed(batch, embedding_weights) for batch in token_batches
        )
        hidden_batches = [embedding.hidden_states for embedding in embeddings]
        attention_masks = tuple(embedding.attention_mask for embedding in embeddings)
        position_ids = tuple(embedding.position_ids for embedding in embeddings)
        del embedding_weights, embeddings

        collected_cache = [] if collect_kv_cache else None
        pending_write = None
        for layer in range(self.config.num_hidden_layers):
            self._log(f"layer={layer} load weights")
            weights = self.weight_loader.load_layer(layer)
            layer_keys = [] if collected_cache is not None else None
            layer_values = [] if collected_cache is not None else None

            for microbatch, inputs in enumerate(hidden_batches):
                self._log(
                    f"layer={layer} microbatch={microbatch} Attention-start"
                )
                attention_result = self.operations.prefill_attention(
                    inputs,
                    attention_masks[microbatch],
                    position_ids[microbatch],
                    weights,
                )
                current_write = self.kv_writer.submit_prefill_microbatch(
                    microbatch,
                    layer,
                    attention_result.keys,
                    attention_result.values,
                )
                if pending_write is not None:
                    self.kv_writer.wait_prefill(pending_write)
                pending_write = current_write

                if collected_cache is not None:
                    layer_keys.append(attention_result.keys)
                    layer_values.append(attention_result.values)
                hidden_batches[microbatch] = self.operations.run_mlp(
                    attention_result.mlp_inputs,
                    weights.mlp,
                    residual=attention_result.hidden_states,
                )
                self._log(
                    f"layer={layer} microbatch={microbatch} complete hidden="
                    f"{tuple(hidden_batches[microbatch].shape)}"
                )
                del attention_result

            if collected_cache is not None:
                collected_cache.append(
                    (
                        torch.cat(layer_keys, dim=1),
                        torch.cat(layer_values, dim=1),
                    )
                )
            del weights

        if pending_write is not None:
            self.kv_writer.wait_prefill(pending_write)

        if last_token_only:
            hidden_states = torch.cat(
                [hidden[:, -1:, :] for hidden in hidden_batches],
                dim=0,
            )
        else:
            hidden_states = torch.cat(hidden_batches, dim=0)
        output = self.operations.run_output_head(
            hidden_states,
            self.weight_loader.load_final_norm(),
            self.weight_loader.load_lm_head(),
            do_sample,
            temperature,
        )
        self._log(
            f"multi-batch output hidden={tuple(hidden_states.shape)}, "
            f"logits={tuple(output.logits.shape)}"
        )
        return ModelPrefillResult(
            hidden_states=hidden_states,
            logits=output.logits,
            next_token_ids=output.next_token_ids,
            kv_cache=tuple(collected_cache) if collected_cache is not None else None,
        )
