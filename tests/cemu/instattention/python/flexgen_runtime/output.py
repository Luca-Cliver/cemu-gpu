from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexGenOutputHeadResult:
    logits: torch.Tensor
    next_token_ids: torch.Tensor


def _rms_norm(
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    normalized = hidden_states.to(torch.float32)
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


def run_flexgen_output_head(
    hidden_states: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    epsilon: float,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> FlexGenOutputHeadResult:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    hidden = _rms_norm(final_norm_weight, hidden_states, epsilon)
    logits = F.linear(hidden, lm_head_weight, bias=None)
    last_token_logits = logits[:, -1, :]

    if do_sample and temperature >= 1e-5:
        probabilities = torch.softmax(last_token_logits / temperature, dim=-1)
        next_token_ids = torch.multinomial(probabilities, num_samples=1)
    else:
        next_token_ids = last_token_logits.argmax(dim=1, keepdim=True)
    return FlexGenOutputHeadResult(logits=logits, next_token_ids=next_token_ids)
