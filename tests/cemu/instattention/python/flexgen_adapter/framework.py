import numpy as np

from .framework_import import load_framework
from .framework_cache import FrameworkSession
from .framework_weights import verify_weights


def create_model(config, env, weight_directory, policy, backends, profiler=None):
    flexgen = load_framework()
    from .framework_attention import CemuSelfAttention

    if (not policy.sep_layer or policy.act_gpu_percent != 100 or policy.compress_weight or
            policy.compress_cache or policy.cpu_cache_compute or policy.attn_sparsity != 1):
        raise ValueError("CEMU FlexGen requires split layers, GPU activations, uncompressed weights/KV; SparF is device-owned")
    if len(backends) != policy.num_gpu_batches:
        raise ValueError("one CEMU backend view is required per GPU microbatch")
    for backend in backends:
        layout = backend.layout.config
        if (layout.batch_size != policy.gpu_batch_size or layout.num_layers != config.num_hidden_layers or
                layout.num_kv_heads != config.n_head or layout.head_dim != config.input_dim // config.n_head or
                layout.dtype != np.dtype(config.dtype)):
            raise ValueError("FlexGen model and CEMU KV geometry do not match")
    verify_weights(config, weight_directory)
    session = FrameworkSession(backends, profiler, config, policy)
    try:
        model = flexgen.OptLM(config, env, str(weight_directory), policy,
            attention_factory=lambda config, env, policy, layer: CemuSelfAttention(config, env, policy, layer, session))
    except BaseException:
        session.close()
        raise
    model.cemu_session = session
    return model


def validate_inputs(model, inputs, decode_steps):
    inputs = np.asarray(inputs)
    if (type(decode_steps) is not int or decode_steps < 1 or inputs.ndim != 2 or
            not np.issubdtype(inputs.dtype, np.integer) or inputs.shape[1] < 1 or
            inputs.shape[0] != model.policy.gpu_batch_size * model.policy.num_gpu_batches):
        raise ValueError("expected a rectangular integer prompt and positive Decode count")
    if np.any(inputs < 0) or np.any(inputs >= model.config.vocab_size) or np.any(inputs == model.config.pad_token_id):
        raise ValueError("initial FlexGen CEMU path requires valid, unpadded token IDs")
    maximum = inputs.shape[1] + decode_steps
    if maximum > model.config.max_seq_len:
        raise ValueError("prompt + Decode exceeds positional embedding capacity")
    if any(maximum > backend.layout.config.max_seq_len for backend in model.cemu_session.backends):
        raise ValueError("prompt + Decode exceeds registered KV capacity")
    return inputs
