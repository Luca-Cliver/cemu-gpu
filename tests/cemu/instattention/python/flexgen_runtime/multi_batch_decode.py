from typing import Any, Callable, Optional, Sequence

import torch

from .decode import (
    FlexGenDecodeAttentionOutput,
    finish_flexgen_decode_attention,
    prepare_flexgen_decode_attention,
)
from .decode_runner import FlexGenDecodeResult, FlexGenDecodeRunner
from .embedding import run_flexgen_embedding
from .mlp import run_flexgen_mlp
from .model_config import FlexGenLlamaConfig
from .output import run_flexgen_output_head
from .weight_prefetch import FlexGenWeightPrefetcher
from .weights import FlexGenWeightLoader


class FlexGenMultiBatchDecodeRunner(FlexGenDecodeRunner):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        attention_backends: Sequence[Any],
        gpu_batch_size: int,
        logger: Optional[Callable[[str], None]] = None,
    ):
        backends = tuple(attention_backends)
        if not backends:
            raise ValueError("attention_backends must not be empty")
        if (
            not isinstance(gpu_batch_size, int)
            or isinstance(gpu_batch_size, bool)
            or gpu_batch_size <= 0
        ):
            raise ValueError("gpu_batch_size must be a positive integer")
        super().__init__(config, weight_loader, backends[0], logger=logger)
        for backend in backends[1:]:
            for method_name in ("append_decode", "decode"):
                if not hasattr(backend, method_name):
                    raise TypeError(
                        f"attention backend must provide {method_name}()"
                    )
        pipeline_modes = tuple(
            bool(getattr(backend, "supports_pipelined_decode", False))
            for backend in backends
        )
        if any(pipeline_modes) and not all(pipeline_modes):
            raise ValueError(
                "all microbatch Attention backends must use the same pipeline mode"
            )
        self.attention_backends = backends
        self.gpu_batch_size = gpu_batch_size
        self.num_gpu_batches = len(backends)
        self.supports_pipelined_decode = all(pipeline_modes)

    @property
    def batch_size(self) -> int:
        return self.gpu_batch_size * self.num_gpu_batches

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
        if token_ids.shape[0] != self.batch_size:
            raise ValueError(
                f"Decode batch size {token_ids.shape[0]} does not match "
                f"{self.num_gpu_batches} GPU batches of {self.gpu_batch_size}"
            )
        if not isinstance(token_position, int) or isinstance(token_position, bool):
            raise TypeError("token_position must be an integer")
        if token_position < 0:
            raise ValueError("token_position must be non-negative")

        token_ids = token_ids.to(self.weight_loader.device)
        start_message = (
            f"multi-batch start position={token_position}, "
            f"gpu_batches={self.num_gpu_batches}, "
            f"gpu_batch_size={self.gpu_batch_size}"
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
        del embedding_weight
        microbatch_hidden = list(
            hidden_states.split(self.gpu_batch_size, dim=0)
        )
        collected_outputs = [] if collect_layer_outputs else None
        valid_tokens = token_position + 1
        current_prefetch = None

        with FlexGenWeightPrefetcher(
            self.weight_loader,
            logger=self.logger,
        ) as weight_prefetcher:
            weight_request = weight_prefetcher.prefetch(0)
            if self.supports_pipelined_decode:
                current_prefetch = self.attention_backends[0].prefetch_decode(
                    0,
                    token_position,
                )
                self._log(
                    "multi-batch prefetch layer=0, microbatch=0, "
                    f"history_tokens={token_position}"
                )

            for layer in range(self.config.num_hidden_layers):
                weights = weight_prefetcher.wait(weight_request)
                next_weight_request = (
                    weight_prefetcher.prefetch(layer + 1)
                    if layer + 1 < self.config.num_hidden_layers
                    else None
                )
                layer_outputs = [None] * self.num_gpu_batches
                pending = None

                for microbatch, backend in enumerate(self.attention_backends):
                    next_location = self._next_location(layer, microbatch)
                    next_prefetch = None
                    if self.supports_pipelined_decode and next_location is not None:
                        next_layer, next_microbatch = next_location
                        next_prefetch = self.attention_backends[
                            next_microbatch
                        ].prefetch_decode(next_layer, token_position)
                        self._log(
                            f"pipeline Load(next) layer={next_layer}, "
                            f"microbatch={next_microbatch}, before "
                            f"Compute(current) layer={layer}, microbatch={microbatch}"
                        )
                    inputs = microbatch_hidden[microbatch]
                    self._log(
                        f"QKV-start position={token_position}, layer={layer}, "
                        f"microbatch={microbatch}"
                    )
                    projection = prepare_flexgen_decode_attention(
                        inputs=inputs,
                        weights=weights.attention,
                        num_heads=self.config.num_attention_heads,
                        num_key_value_heads=self.config.num_key_value_heads,
                        token_position=token_position,
                        rope_theta=self.config.rope_theta,
                        epsilon=self.config.rms_norm_epsilon,
                    )
                    self._log(
                        f"QKV-complete position={token_position}, layer={layer}, "
                        f"microbatch={microbatch}"
                    )
                    self._log(
                        f"layer={layer} microbatch={microbatch} projected "
                        f"Q={tuple(projection.query.shape)}, "
                        f"K/V={tuple(projection.key.shape)}"
                    )

                    if self.supports_pipelined_decode:
                        request = backend.submit_prefetched_decode(
                            prefetch=current_prefetch,
                            layer=layer,
                            token=token_position,
                            query=projection.query,
                            key=projection.key,
                            value=projection.value,
                            valid_tokens=valid_tokens,
                        )
                        if pending is not None:
                            self._finish_pending(
                                pending,
                                weights,
                                microbatch_hidden,
                                layer_outputs,
                            )
                        pending = (
                            microbatch,
                            backend,
                            request,
                            inputs,
                            projection,
                        )
                        current_prefetch = next_prefetch
                    else:
                        backend.append_decode(
                            layer,
                            token_position,
                            projection.key,
                            projection.value,
                        )
                        attention_output = backend.decode(
                            layer=layer,
                            query=projection.query,
                            valid_tokens=valid_tokens,
                        )
                        self._finish_microbatch(
                            microbatch,
                            inputs,
                            projection,
                            attention_output,
                            weights,
                            microbatch_hidden,
                            layer_outputs,
                        )

                if pending is not None:
                    self._finish_pending(
                        pending,
                        weights,
                        microbatch_hidden,
                        layer_outputs,
                    )
                if collected_outputs is not None:
                    collected_outputs.append(self._merge_layer_outputs(layer_outputs))
                del weights
                weight_request = next_weight_request

        hidden_states = torch.cat(microbatch_hidden, dim=0)
        output = run_flexgen_output_head(
            hidden_states=hidden_states,
            final_norm_weight=self.weight_loader.load_final_norm(),
            lm_head_weight=self.weight_loader.load_lm_head(),
            epsilon=self.config.rms_norm_epsilon,
            do_sample=do_sample,
            temperature=temperature,
        )
        output_message = f"multi-batch output logits={tuple(output.logits.shape)}"
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

    def _next_location(self, layer: int, microbatch: int):
        if microbatch + 1 < self.num_gpu_batches:
            return layer, microbatch + 1
        if layer + 1 < self.config.num_hidden_layers:
            return layer + 1, 0
        return None

    def _finish_pending(
        self,
        pending,
        weights,
        microbatch_hidden,
        layer_outputs,
    ) -> None:
        microbatch, backend, request, inputs, projection = pending
        attention_output = backend.wait_decode(request)
        self._finish_microbatch(
            microbatch,
            inputs,
            projection,
            attention_output,
            weights,
            microbatch_hidden,
            layer_outputs,
        )

    def _finish_microbatch(
        self,
        microbatch,
        inputs,
        projection,
        attention_output,
        weights,
        microbatch_hidden,
        layer_outputs,
    ) -> None:
        attention_result = finish_flexgen_decode_attention(
            inputs=inputs,
            projection=projection,
            attention_output=attention_output,
            weights=weights.attention,
            epsilon=self.config.rms_norm_epsilon,
        )
        self._log(f"MLP-start microbatch={microbatch}")
        microbatch_hidden[microbatch] = run_flexgen_mlp(
            attention_result.mlp_inputs,
            weights.mlp,
            residual=attention_result.hidden_states,
        )
        self._log(f"MLP-complete microbatch={microbatch}")
        layer_outputs[microbatch] = attention_result
        self._log(
            f"microbatch={microbatch} attention complete, "
            f"hidden={tuple(microbatch_hidden[microbatch].shape)}"
        )

    @staticmethod
    def _merge_layer_outputs(outputs) -> FlexGenDecodeAttentionOutput:
        if any(output is None for output in outputs):
            raise RuntimeError("not every microbatch produced an Attention output")
        return FlexGenDecodeAttentionOutput(
            hidden_states=torch.cat(
                [output.hidden_states for output in outputs],
                dim=0,
            ),
            mlp_inputs=torch.cat(
                [output.mlp_inputs for output in outputs],
                dim=0,
            ),
            query=torch.cat([output.query for output in outputs], dim=0),
            key=torch.cat([output.key for output in outputs], dim=1),
            value=torch.cat([output.value for output in outputs], dim=1),
            attention_output=torch.cat(
                [output.attention_output for output in outputs],
                dim=0,
            ),
        )
