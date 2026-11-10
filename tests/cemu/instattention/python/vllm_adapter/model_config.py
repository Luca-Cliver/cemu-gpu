"""Model geometry shared by the offline launcher and the vLLM worker."""

from collections.abc import Mapping


def model_geometry(config):
    get = config.get if isinstance(config, Mapping) else (
        lambda name, default=None: getattr(config, name, default))
    kind = get("model_type")
    if kind not in ("llama", "opt"):
        raise ValueError("CEMU supports Llama and OPT models only")
    layers, heads, hidden = (get(name) for name in (
        "num_hidden_layers", "num_attention_heads", "hidden_size"))
    if any(type(value) is not int or value <= 0 for value in (layers, heads, hidden)):
        raise ValueError("invalid model layer/head geometry")
    # OPTAttention uses MHA and hidden_size // num_attention_heads. Its config
    # does not define num_key_value_heads, and it does not use Llama's RoPE.
    if kind == "opt" and hidden % heads:
        raise ValueError("OPT hidden size must be divisible by its head count")
    dim = hidden // heads if kind == "opt" else (get("head_dim") or hidden // heads)
    kv_heads = heads if kind == "opt" else get("num_key_value_heads", heads)
    if type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("invalid KV head grouping")
    if dim != 128:
        raise ValueError(f"CEMU currently requires head_dim=128, got {dim}")
    # Both local vLLM model implementations pass unscaled Q to Attention.
    # Old FlexGen's scale=1 default must not be used for vLLM OPT.
    return layers, heads, kv_heads, dim, dim**-0.5
