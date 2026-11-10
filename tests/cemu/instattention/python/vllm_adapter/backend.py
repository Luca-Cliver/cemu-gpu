"""Dense GPU prefill; device-owned KV and autonomous SparF decode."""

from dataclasses import dataclass
import re

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder, AttentionType,
)

from .worker import runtime
from .profiling import batch_phase
from phase_profiler import profile_scope, profile_tensor_scope, stack_profiler, profile_request


class CemuAttentionBackend(AttentionBackend):
    supported_dtypes = [torch.float16]
    supported_kv_cache_dtypes = ["auto", "float16"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        return "CEMU"

    @staticmethod
    def get_impl_cls():
        return CemuAttentionImpl

    @staticmethod
    def get_builder_cls():
        return CemuAttentionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls):
        return [128]

    @classmethod
    def supports_kv_connector(cls):
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return False


@dataclass(frozen=True)
class Metadata:
    work: tuple | None


class CemuAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        r = runtime()
        for name in layer_names:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None or not 0 <= int(match[1]) < r.session.layers:
                raise ValueError(f"unrecognized Attention layer: {name}")
            r.layer_ids[name] = int(match[1])

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        r = runtime()
        if r.work is None:
            if r.active:
                raise RuntimeError("real CEMU step missing request metadata")
            return Metadata(None)
        starts = common_attn_metadata.query_start_loc_cpu.tolist()
        expected = [w.offset for w in r.work]
        expected.append(sum(w.count for w in r.work))
        if common_prefix_len or starts != expected:
            raise ValueError("CEMU request/token mapping disagrees with vLLM metadata")
        return Metadata(r.work)


class Staging:
    """Reusable pinned buffers shared across sequential layers in one worker."""

    def __init__(self, tokens, requests, kv_heads, query_heads, dim):
        self.keys = torch.empty((tokens, kv_heads, dim), dtype=torch.float16, pin_memory=True)
        self.values = torch.empty_like(self.keys, pin_memory=True)
        self.queries = torch.empty((requests, query_heads, dim), dtype=torch.float16, pin_memory=True)
        # Page alignment lets C++ pread directly into this pinned buffer. Keep
        # the backing storage alive through the tensor view and its NumPy view.
        result_bytes = requests * query_heads * dim * 2
        storage = torch.empty(result_bytes + 4095, dtype=torch.uint8, pin_memory=True)
        offset = -storage.data_ptr() % 4096
        self.results = storage[offset:offset + result_bytes].view(torch.float16).view(
            requests, query_heads, dim)
        self.key_array, self.value_array = self.keys.numpy(), self.values.numpy()
        self.query_array, self.result_array = self.queries.numpy(), self.results.numpy()
        self.ready = torch.cuda.Event()
        self.reusable = torch.cuda.Event()
        self.pending = False

    def wait(self):
        if self.pending:
            self.reusable.synchronize()
            self.pending = False


class CemuAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **extra):
        if (alibi_slopes is not None or sliding_window is not None or
                logits_soft_cap not in (None, 0) or attn_type != AttentionType.DECODER or
                kv_sharing_target_layer_name is not None or extra):
            raise ValueError("CEMU supports plain causal Llama/OPT Attention only")
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        r = runtime()
        s = r.session
        if (num_heads, self.num_kv_heads, head_size, scale) != (
                s.query_heads, s.kv_heads, s.head_dim, s.scale):
            raise ValueError("Attention geometry/scale disagrees with CEMU session")

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        profiler = runtime().session.profiler
        work = attn_metadata.work if attn_metadata is not None else None
        if profiler is None or work is None:
            return self._forward(layer, query, key, value, kv_cache, attn_metadata,
                                 output, output_scale, output_block_scale)
        phase = batch_phase(w.prefill for w in work)
        context = None
        if profiler.interval_path:
            context = runtime().session.profile_context(work, runtime().layer_ids[layer.layer_name])
        with profile_request(profiler, context), profile_scope(profiler, phase + ".attention_host_wall"):
            return self._forward(layer, query, key, value, kv_cache, attn_metadata,
                                 output, output_scale, output_block_scale)

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                 output, output_scale=None, output_block_scale=None):
        r = runtime()
        if attn_metadata is None or attn_metadata.work is None:
            if r.active:
                raise RuntimeError("real Attention call has no CEMU request identity")
            output.zero_()  # vLLM allocation/profile probe: no device I/O.
            return output
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("CEMU does not support quantized output")
        work = attn_metadata.work
        profiler = r.session.profiler
        detailed = stack_profiler(profiler)
        phase = batch_phase(w.prefill for w in work) if profiler is not None else ""
        n = sum(w.count for w in work)
        if query.dtype != torch.float16 or key is None or value is None:
            raise ValueError("CEMU requires FP16 Q and unfused K/V inputs")
        if not query.is_cuda:
            raise ValueError("vLLM CEMU prefill requires a Guest CUDA GPU")
        if r.staging is None:
            # Grow once to configured maxima, never allocate per layer/token.
            with profile_scope(profiler, "staging.allocate_wall"):
                r.staging = Staging(r.max_batch_tokens, r.max_requests,
                                    self.num_kv_heads, self.num_heads, self.head_size)
        stage = r.staging
        decodes = [w for w in work if not w.prefill]
        with profile_scope(profiler, phase + ".staging_reuse_wait_wall"):
            stage.wait()
        with profile_scope(profiler, phase + ".d2h_enqueue_wall"), \
                profile_tensor_scope(detailed, phase + ".d2h", query.device):
            stage.keys[:n].copy_(key[:n], non_blocking=True)
            stage.values[:n].copy_(value[:n], non_blocking=True)
            if len(decodes) == len(work):
                stage.queries[:len(decodes)].copy_(query[:n], non_blocking=True)
            else:
                for i, w in enumerate(decodes):
                    stage.queries[i].copy_(query[w.offset], non_blocking=True)
            stage.ready.record()
        for w in work:
            if w.prefill:
                sl = slice(w.offset, w.offset + w.count)
                # Require Flash SDPA: no accidental quadratic math fallback.
                with profile_tensor_scope(detailed, "prefill.dense_attention", query.device), \
                        sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = scaled_dot_product_attention(
                        query[sl].transpose(0, 1)[None],
                        key[sl].transpose(0, 1)[None], value[sl].transpose(0, 1)[None],
                        is_causal=True, scale=self.scale,
                        enable_gqa=self.num_heads != self.num_kv_heads)
                output[sl].copy_(result[0].transpose(0, 1))
        with profile_scope(profiler, phase + ".d2h_wait_wall"):
            stage.ready.synchronize()  # Existing D2H wait, not an extra profiling sync.
        keys, values, queries = stage.key_array, stage.value_array, stage.query_array
        layer_id = r.layer_ids[layer.layer_name]
        try:
            # Accept all writes before readers, avoiding append/read lock ping-pong.
            for w in work:
                sl = slice(w.offset, w.offset + w.count)
                r.session.append(w, layer_id, keys[sl], values[sl])
            result_array = stage.result_array
            if decodes:
                r.session.decode_batch(decodes, layer_id, queries[:len(decodes)],
                                       output=result_array[:len(decodes)])
        except BaseException as error:
            r.session.failure = error
            raise
        if decodes:
            with profile_scope(profiler, phase + ".h2d_enqueue_wall"), \
                    profile_tensor_scope(detailed, phase + ".h2d", query.device):
                if len(decodes) == len(work):
                    output[:n].copy_(stage.results[:n], non_blocking=True)
                else:
                    for i, w in enumerate(decodes):
                        output[w.offset].copy_(stage.results[i], non_blocking=True)
                stage.reusable.record()
                stage.pending = True
        return output
