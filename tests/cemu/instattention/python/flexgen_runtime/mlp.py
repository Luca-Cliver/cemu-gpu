import torch
import torch.nn.functional as F

from .weights import FlexGenMlpWeights


def run_flexgen_mlp(
    inputs: torch.Tensor,
    weights: FlexGenMlpWeights,
) -> torch.Tensor:
    if inputs.ndim != 3:
        raise ValueError("MLP inputs must have shape [batch, sequence, hidden]")
    if not isinstance(weights, FlexGenMlpWeights):
        raise TypeError("weights must be FlexGenMlpWeights")

    gate = F.silu(F.linear(inputs, weights.gate, bias=None))
    up = F.linear(inputs, weights.up, bias=None)
    output = F.linear(gate * up, weights.down, bias=None)
    return inputs + output
