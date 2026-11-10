"""CEMU Attention plugin for the pinned vLLM V1 runner."""


def register():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    path = "vllm_adapter.backend.CemuAttentionBackend"
    if "CEMU" in AttentionBackendEnum.__members__:
        if AttentionBackendEnum["CEMU"].get_path() != path:
            raise RuntimeError("another plugin already registered the CEMU backend")
        return
    register_backend("CEMU", path)
