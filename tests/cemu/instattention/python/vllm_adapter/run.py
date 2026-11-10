"""Offline Llama/OPT functional and performance tests inside the CEMU Guest."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from .session import Config, load_runtime
from .model_config import model_geometry
from .profiling import device_report, print_timing
from .guest_clock import configure_guest_clock, verify_guest_clock


def generate_requests(llm, prompts, sampling_params, *, fixed_batch_size=None):
    if fixed_batch_size is None:
        return llm.generate(prompts, sampling_params)
    if len(prompts) != fixed_batch_size:
        raise ValueError("fixed-batch prompt count does not match fixed_batch_size")
    core_client = llm.llm_engine.engine_core
    utility = getattr(core_client, "call_utility", None)
    if not callable(utility) or not callable(getattr(llm, "enqueue", None)) or not callable(
            getattr(llm, "wait_for_completion", None)):
        raise RuntimeError("fixed-batch admission requires the pinned vLLM synchronous multiprocess client")
    utility("pause_scheduler", "keep", False)
    request_ids = llm.enqueue(prompts, sampling_params)
    if len(request_ids) != fixed_batch_size or len(set(request_ids)) != fixed_batch_size:
        raise RuntimeError("fixed-batch enqueue did not admit all distinct requests; scheduler remains paused")
    print(f"[vllm-cemu] admission barrier: {fixed_batch_size} requests queued; resuming scheduler", flush=True)
    utility("resume_scheduler")
    return llm.wait_for_completion()


def drain_session(llm, *, failed):
    try:
        return llm.collective_rpc("cemu_drain")
    except Exception as error:
        if not failed:
            raise
        print(f"[vllm-cemu] cleanup failed after generation error: {error}; "
              "preserving original exception", file=sys.stderr, flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-token-ids", help="JSON array of token-ID arrays")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int,
                        help="scheduler token budget; defaults to capacity times maximum sequence length")
    parser.add_argument("--native", action="store_true", help="dense vLLM baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("off", "basic", "detailed"), default="basic",
                        help="basic wall timers; detailed adds C++ splits and deferred CUDA events")
    parser.add_argument("--device-timeline", default="/root/cemu-logs/device-timeline.jsonl")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    if args.profile == "detailed" and not args.native:
        import _cemu_client
        if getattr(_cemu_client, "IO_PROFILE_VERSION", 0) != 1:
            parser.error("detailed profiling requires rebuilding the CEMU Guest client")
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file() or any(model.rglob("*.incomplete")):
        parser.error("model must be a complete local snapshot")
    prompts = args.prompt or ["The meaning of life is"]
    if args.prompt_token_ids:
        prompts = [{"prompt_token_ids": ids} for ids in json.loads(
            Path(args.prompt_token_ids).read_text())]
    if not prompts or len(prompts) > cfg.request_capacity:
        parser.error("number of requests exceeds session request_capacity")
    if cfg.fixed_batch_size is not None and (
            not args.prompt_token_ids or len(prompts) != cfg.fixed_batch_size or
            len({len(prompt["prompt_token_ids"]) for prompt in prompts}) != 1):
        parser.error("fixed-batch experiments require fixed_batch_size equal-length explicit prompts")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    token_budget = args.max_num_batched_tokens or cfg.max_tokens * len(prompts)
    if ((args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0) or
            token_budget < cfg.max_tokens):
        parser.error("token budget must cover max_model_len with chunked prefill disabled")
    if cfg.fixed_batch_size is not None and token_budget < sum(
            len(prompt["prompt_token_ids"]) for prompt in prompts):
        parser.error("fixed-batch token budget must cover the whole Prefill batch")
    hf = json.loads((model / "config.json").read_text())
    try:
        layers, heads, kv_heads, head_dim, _scale = model_geometry(hf)
        runtime_model = load_runtime(cfg)
        if runtime_model is not None:
            if not args.prompt_token_ids:
                raise ValueError("runtime experiments require explicit prompt token IDs for shape validation")
            lengths = {len(prompt["prompt_token_ids"]) for prompt in prompts}
            if len(lengths) != 1:
                raise ValueError("uniform ratio tables cannot calibrate unequal-length batches")
            prompt_length = next(iter(lengths))
            if not prompt_length or prompt_length + args.max_tokens > cfg.max_tokens:
                raise ValueError("prompt plus output exceeds max_model_len")
            if token_budget < prompt_length * len(prompts):
                raise ValueError("runtime experiment requires a token budget covering the whole Prefill batch")
            if args.max_tokens > 1:
                runtime_model.softmax_ratios.require_sparf(
                    len(prompts), heads, prompt_length, args.max_tokens - 1, cfg.compression_ratio)
    except ValueError as error:
        parser.error(str(error))
    if not (list(model.glob("*.safetensors")) or list(model.glob("*.bin"))):
        parser.error("local model has no weights")
    for index in model.glob("*.index.json"):
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if any(not (model / shard).is_file() for shard in shards):
            parser.error(f"model snapshot is missing shards listed in {index.name}")
    try:
        guest_clock = configure_guest_clock()
    except (OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"[vllm-clock] {guest_clock['before']} -> hpet; "
          "Guest timers exclude functional freeze; CUDA events are diagnostic only", flush=True)
    # Bound the compatibility GPU cache instead of consuming all spare VRAM.
    # One null block plus a full maximum prefix per concurrent request.
    block_size = 16
    blocks = (cfg.max_tokens + block_size - 1) // block_size * len(prompts) + 1
    cache_bytes = blocks * block_size * layers * kv_heads * head_dim * 4
    os.environ["CEMU_VLLM_CONFIG"] = str(Path(args.config).resolve())
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CEMU_VLLM_PROFILE"] = args.profile
    # The Guest uses precompiled vLLM extensions without an nvcc toolchain.
    # Startup profiles random sampling even for this greedy smoke test.
    # Use the built-in sampler for both the native and CEMU runs by default.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from . import register
    register()

    kwargs = {} if args.native else dict(
        worker_cls="vllm_adapter.worker.CemuWorker",
        attention_config={"backend": "CEMU"})
    if args.native and args.profile != "off":
        kwargs["worker_cls"] = "vllm_adapter.worker.ProfiledWorker"
    initialize_started = time.perf_counter()
    llm = LLM(
        model=str(model), dtype="half", max_model_len=cfg.max_tokens,
        max_num_seqs=min(len(prompts), cfg.request_capacity),
        max_num_batched_tokens=token_budget, enforce_eager=True,
        block_size=block_size, kv_cache_memory_bytes=cache_bytes,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, seed=0, **kwargs)
    initialize_seconds = time.perf_counter() - initialize_started
    verify_guest_clock()
    timeline = Path(args.device_timeline)
    cursor = None
    if not args.native and args.profile != "off":
        try:
            stat = timeline.stat()
            cursor = (stat.st_dev, stat.st_ino, stat.st_size)
        except FileNotFoundError:
            # QEMU opens this log lazily on the first actual EXEC.
            cursor = (None, None, 0)
        except OSError:
            pass
    started = time.perf_counter()
    drain = None
    try:
        results = generate_requests(llm, prompts, SamplingParams(
            temperature=0, max_tokens=args.max_tokens, ignore_eos=True, logprobs=5),
            fixed_batch_size=cfg.fixed_batch_size)
        generate_seconds = time.perf_counter() - started
    finally:
        drain_started = time.perf_counter()
        if not args.native:
            drain = drain_session(llm, failed=sys.exc_info()[0] is not None)
        drain_seconds = time.perf_counter() - drain_started
    generate_and_drain_seconds = time.perf_counter() - started
    verify_guest_clock()
    guest_clock["after"] = "hpet"
    timing = {"enabled": False}
    if args.profile != "off":
        timing = (llm.collective_rpc("cemu_timing")[0] if args.native else drain[0]["timing"])
    device_timing = {"available": False, "reason": "native run or profiling disabled"}
    if not args.native and args.profile != "off":
        device_timing = {"available": False, "reason": "device timeline unavailable before generate"}
        if cursor is not None:
            try:
                with timeline.open("rb") as source:
                    stat = os.fstat(source.fileno())
                    if ((cursor[0] is not None and (stat.st_dev, stat.st_ino) != cursor[:2])
                            or stat.st_size < cursor[2]):
                        raise ValueError("device timeline replaced or truncated during generate")
                    source.seek(cursor[2])
                    records = [json.loads(line) for line in source if line.strip()]
                device_timing = device_report(records, drain[0]["program_id"],
                                              drain[0]["attention_exec_count"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                device_timing = {"available": False, "reason": str(error)}
    payload = {
        "backend": "native" if args.native else "CEMU",
        "generate_seconds": generate_seconds,
        "initialize_seconds": initialize_seconds,
        "drain_rpc_seconds": drain_seconds if not args.native else 0.0,
        "generate_and_drain_seconds": generate_and_drain_seconds,
        "timing": timing, "device_timing": device_timing,
        "guest_clock": guest_clock,
        "cemu_config": vars(cfg),
        "max_num_batched_tokens": token_budget,
        "request_admission": "barrier" if cfg.fixed_batch_size is not None else "streaming",
        "runtime_model": "sparf_gemv_filter_softmax_ratio" if runtime_model else "functional_zero_budget",
        "compatibility_gpu_kv_bytes": cache_bytes,
        "drain": drain,
        "outputs": [{"request_id": result.request_id,
                     "prompt_token_ids": result.prompt_token_ids,
                     "token_ids": list(result.outputs[0].token_ids),
                     "text": result.outputs[0].text,
                     "logprobs": [{str(token): lp.logprob for token, lp in step.items()}
                                  for step in result.outputs[0].logprobs]}
                    for result in results],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[timing] initialize={initialize_seconds:.3f}s, generate={generate_seconds:.6f}s, "
          f"terminal_drain_rpc={payload['drain_rpc_seconds']:.6f}s")
    print_timing(timing)
    if device_timing.get("available"):
        ratio = device_timing["device_percent_of_decode_command_window"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}%"
        print(f"[timing] Device commands (QEMU virtual, freeze excluded): "
              f"{device_timing['device_command_union_seconds']:.6f}s, "
              f"{ratio_text} of virtual first-to-last decode EXEC window (not total generate time)")
        print(f"[timing] Device modeled CU seconds: {device_timing['compute_seconds_by_stage']}")
        if device_timing["zero_budget_compute_phases"]:
            print("[timing] Some compute phases have zero modeled budget; "
                  "frozen functional GPU runtime is not a substitute for calibrated compute latency.")
    elif not args.native and args.profile != "off":
        print(f"[timing] Device timeline unavailable: {device_timing['reason']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
