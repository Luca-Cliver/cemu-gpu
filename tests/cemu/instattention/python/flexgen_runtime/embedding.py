import torch
import torch.nn.functional as F


def run_flexgen_embedding(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    pad_token_id: int,
):
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_ids must use an integer dtype")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")

    attention_mask = token_ids.ne(pad_token_id)
    hidden_states = F.embedding(
        token_ids,
        embedding_weight,
        padding_idx=pad_token_id,
    )
    position_ids = torch.arange(
        token_ids.shape[1],
        dtype=torch.long,
        device=token_ids.device,
    ).expand(token_ids.shape[0], -1)
    return hidden_states, attention_mask, position_ids
