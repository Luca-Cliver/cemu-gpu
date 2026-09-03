from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from .decode_runner import FlexGenDecodeResult, FlexGenDecodeRunner


@dataclass(frozen=True)
class FlexGenGenerationStep:
    step: int
    token_position: int
    input_token_ids: torch.Tensor
    decode_result: FlexGenDecodeResult


@dataclass(frozen=True)
class FlexGenGenerationResult:
    token_ids: torch.Tensor
    steps: Tuple[FlexGenGenerationStep, ...]

    @property
    def next_token_ids(self) -> torch.Tensor:
        return self.token_ids[:, -1:]


class FlexGenGenerationRunner:
    def __init__(
        self,
        decode_runner: FlexGenDecodeRunner,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(decode_runner, FlexGenDecodeRunner):
            raise TypeError("decode_runner must be a FlexGenDecodeRunner")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")

        self.decode_runner = decode_runner
        self.logger = logger

    def run(
        self,
        initial_token_ids: torch.Tensor,
        start_position: int,
        decode_steps: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        collect_layer_outputs: bool = False,
    ) -> FlexGenGenerationResult:
        if initial_token_ids.ndim != 2 or initial_token_ids.shape[1] != 1:
            raise ValueError("initial_token_ids must have shape [batch, 1]")
        if not isinstance(start_position, int) or isinstance(start_position, bool):
            raise TypeError("start_position must be an integer")
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        if not isinstance(decode_steps, int) or isinstance(decode_steps, bool):
            raise TypeError("decode_steps must be an integer")
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")

        current_token_ids = initial_token_ids.to(
            self.decode_runner.weight_loader.device
        )
        token_sequence = [current_token_ids.detach().clone()]
        steps = []
        if self.logger is not None:
            self._log(
                f"start position={start_position}, decode_steps={decode_steps}, "
                f"initial_tokens={current_token_ids.detach().cpu().reshape(-1).tolist()}"
            )

        for step in range(decode_steps):
            token_position = start_position + step
            input_token_ids = current_token_ids.detach().clone()
            if self.logger is not None:
                self._log(
                    f"step={step}, position={token_position}, "
                    f"valid_tokens={token_position + 1}, "
                    f"input={input_token_ids.cpu().reshape(-1).tolist()}"
                )
            decode_result = self.decode_runner.run(
                input_token_ids,
                token_position=token_position,
                do_sample=do_sample,
                temperature=temperature,
                collect_layer_outputs=collect_layer_outputs,
            )
            current_token_ids = decode_result.next_token_ids
            token_sequence.append(current_token_ids.detach().clone())
            steps.append(
                FlexGenGenerationStep(
                    step=step,
                    token_position=token_position,
                    input_token_ids=input_token_ids,
                    decode_result=decode_result,
                )
            )
            if self.logger is not None:
                self._log(
                    f"step={step} output="
                    f"{current_token_ids.detach().cpu().reshape(-1).tolist()}"
                )

        generated = torch.cat(token_sequence, dim=1)
        if self.logger is not None:
            self._log(
                f"complete token_ids={tuple(generated.shape)}, "
                f"last_tokens={generated[:, -1].detach().cpu().tolist()}"
            )
        return FlexGenGenerationResult(
            token_ids=generated,
            steps=tuple(steps),
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[generation] {message}")
