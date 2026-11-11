import argparse
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import torch

from flexgen_adapter.framework import create_model, validate_inputs
from flexgen_adapter.framework_environment import make_environment, make_policy
from flexgen_adapter.framework_import import load_framework
from flexgen_adapter.framework_weights import config_from_json, verify_weights
from opt_inputs import (
    comparison_case, compare_tokens, decode_output, load_reference,
    load_tokenizer, prepare_prompt, write_json,
)
from phase_profiler import PhaseProfiler


def arguments():
    parser = argparse.ArgumentParser(description="InstAttention FlexGen OptLM.generate -> CEMU autonomous SparF")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--storage", choices=("fdm", "native"), default="fdm")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=1024)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt-ids", help="JSON rectangular token IDs; otherwise the same synthetic prompt as the old runner")
    inputs.add_argument("--prompt-text", help="local tokenizer encodes one sentence, repeated per batch; actual length replaces --prompt-length")
    parser.add_argument("--decode-text", action="store_true", help="decode output with the local tokenizer, also for --prompt-ids")
    parser.add_argument("--reference-result", help="compare tokens against a same-input result JSON; not a logits accuracy test")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--guest-device", default="cuda:0")
    parser.add_argument("--csd-target", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--program", required=True)
    parser.add_argument("--aggregate-program", default="./build/fdm_kv_aggregate.so")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--native-device", default="/dev/ng0n1")
    parser.add_argument("--native-budget", type=int, default=6 * 1024**3)
    parser.add_argument("--fdm-meta-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--fdm-data-bytes", type=int, default=3968 * 1024**2)
    parser.add_argument("--fdm-input-slots", type=int, choices=(0, 1, 2), default=0,
                        help="0: legacy input; 1/2: tagged synchronous input slots")
    parser.add_argument("--fdm-input-bytes", type=int, default=None,
                        help="capacity of each KV input slot; default fits a whole-layer packet")
    parser.add_argument("--channels-per-page", type=int, default=8)
    parser.add_argument("--write-batch-pages", type=int, default=256)
    parser.add_argument("--attention-slots", type=int, default=2)
    parser.add_argument("--top-r", type=int, default=16)
    parser.add_argument("--compression-ratio", type=int, default=8)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--softmax-ratio-file", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--profile-stack", action="store_true", help="detailed Guest scheduling, packing and client IO intervals; not pure CPU time")
    parser.add_argument("--result-output", required=True)
    args = parser.parse_args()
    if args.storage != "fdm" and (args.fdm_input_slots or args.fdm_input_bytes is not None):
        parser.error("FDM input slot options require --storage fdm")
    return args


def open_backends(stack, args, config, directory, runtime, profiler):
    from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout
    from cemu_flexgen.sparf_group_layout import SparfGroupLayout
    from cemu_flexgen.fdm_kvbuffer_store import FDM_KVbuffer
    from cemu_flexgen.native_kv_store import NativeKvStore
    from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
    from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler
    from flexgen_adapter.sparf_attention import SparfGroupedAttentionBackend

    maximum = args.prompt_length + args.decode_steps
    dimension = config.input_dim // config.n_head
    grouping = SparfGroupLayout(dimension, args.channels_per_page, dtype=config.dtype)
    common = dict(layout=grouping, batches=args.batch_size, layers=config.num_hidden_layers,
        heads=config.n_head, max_tokens=maximum, microbatch_size=args.gpu_batch_size,
        control_path=args.control, namespace_path=args.namespace, batch_pages=args.write_batch_pages)
    if args.storage == "fdm":
        store = FDM_KVbuffer(**common, fdm_directory=directory, meta_bytes=args.fdm_meta_bytes,
            data_bytes=args.fdm_data_bytes, aggregate_program=args.aggregate_program, profiler=profiler,
            input_slots=args.fdm_input_slots, input_bytes=args.fdm_input_bytes)
        print(f"[flexgen-cemu] FDM layer packets: input_bytes={store.input_bytes}, "
              f"input_slots={store.input_slots}, data_bytes={store.data_bytes}, "
              f"batch_count={args.gpu_batch_size}; one write per layer/microbatch; "
              "completion policy follows device binding", flush=True)
    else:
        store = NativeKvStore(**common, device_path=args.native_device, budget=args.native_budget)
    stack.enter_context(store)
    layout = SparfKvCacheLayout(KvLayoutConfig(config.num_hidden_layers, maximum,
        args.gpu_batch_size, config.n_head, dimension, config.dtype))
    slots = tuple(CemuSparfDevice(layout, "unused", "unused", "unused", directory,
        args.program, query_heads=config.n_head, top_r=args.top_r,
        compression_ratio=args.compression_ratio, control_path=args.control,
        namespace_path=args.namespace, cuda_target=args.csd_target == "cuda",
        runtime_model=runtime, profiler=profiler, name_prefix=f"flexgen_slot{slot}",
        autonomous=True, grouped_store=store) for slot in range(args.attention_slots))
    scheduler = CemuSparfSlotScheduler(slots)
    heads_per_view = config.num_hidden_layers * args.gpu_batch_size * config.n_head
    backends = tuple(stack.enter_context(SparfGroupedAttentionBackend(layout, store, scheduler,
        microbatch * heads_per_view, profiler=profiler))
        for microbatch in range(args.batch_size // args.gpu_batch_size))
    return store, backends


def main():
    args = arguments()
    if args.profile_stack:
        import _cemu_client
        if getattr(_cemu_client, "IO_PROFILE_VERSION", 0) != 1:
            raise RuntimeError("--profile-stack requires rebuilding the Guest client with make instattention")
    if args.prompt_ids and Path(args.prompt_ids).resolve() == Path(args.result_output).resolve():
        raise ValueError("result output must not overwrite prompt IDs")
    weight_logger = logging.getLogger("flexgen.weights")
    weight_logger.setLevel(logging.INFO)
    weight_logger.propagate = False
    weight_handler = logging.StreamHandler(sys.stdout)
    weight_handler.setFormatter(logging.Formatter("[flexgen-cemu] %(message)s"))
    weight_logger.addHandler(weight_handler)
    config = config_from_json(Path(args.model_dir) / "config.json")
    print(f"[flexgen-cemu] checking converted weights: {args.weights_dir}", flush=True)
    verify_weights(config, args.weights_dir)
    print("[flexgen-cemu] converted weight files checked; GPU preload follows backend setup", flush=True)
    if (min(args.batch_size, args.gpu_batch_size, args.prompt_length, args.decode_steps,
            args.top_r, args.compression_ratio) <= 0 or args.attention_slots < 2 or
            args.batch_size % args.gpu_batch_size):
        raise ValueError("positive sizes, divisible microbatches, and at least two Attention slots required")
    prompt, tokenizer = prepare_prompt(args.model_dir, args.batch_size, args.prompt_length,
                                       args.decode_steps, args.prompt_ids, args.prompt_text)
    args.prompt_length = prompt.shape[1]
    if args.decode_text and tokenizer is None:
        tokenizer = load_tokenizer(args.model_dir)
    case = comparison_case(args.model_dir, prompt, args.decode_steps, args.top_r, args.compression_ratio)
    reference = None
    if args.reference_result:
        if Path(args.reference_result).resolve() == Path(args.result_output).resolve():
            raise ValueError("result output must not overwrite the reference result")
        reference = load_reference(args.reference_result, case)
    prompt_output = Path(args.result_output).with_suffix(".prompt.json")
    if args.reference_result and prompt_output.resolve() == Path(args.reference_result).resolve():
        raise ValueError("prompt output must not overwrite the reference result")
    write_json(prompt_output, prompt.tolist())
    print(f"[flexgen-cemu] fixed input saved: {prompt_output}; actual prompt length={args.prompt_length}; "
          f"first IDs={prompt[0, :16].tolist()}", flush=True)
    if not torch.cuda.is_available() or torch.device(args.guest_device).type != "cuda":
        raise RuntimeError("the full inference baseline requires a Guest CUDA GPU")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments import SparfAttentionRuntimeModel, load_experiment_config
    from experiments.softmax_ratio import SoftmaxRatioTable

    ratios = SoftmaxRatioTable.load(args.softmax_ratio_file)
    ratios.require_sparf(args.gpu_batch_size, config.n_head, args.prompt_length,
                         args.decode_steps, args.compression_ratio)
    runtime = SparfAttentionRuntimeModel(load_experiment_config(args.runtime_config).instcsd, softmax_ratios=ratios)
    clock_path = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
    clock = clock_path.read_text().strip() if clock_path.exists() else "unknown"
    framework = load_framework()
    print(f"[flexgen-cemu] framework={framework.__file__}; storage={args.storage}; I/O={os.getenv('CEMU_IO_BACKEND', 'ioctl')}", flush=True)
    print(f"[flexgen-cemu] clocksource={clock}; Guest clock intervals are not CUDA event time; use validated HPET for freeze-aware comparisons", flush=True)
    print(f"[flexgen-cemu] batch={args.batch_size}, microbatch={args.gpu_batch_size}, prompt={args.prompt_length}, Decode={args.decode_steps}, max_new_tokens={args.decode_steps + 1}", flush=True)
    profiler = PhaseProfiler(stack=args.profile_stack)
    env = make_environment(args.guest_device, config.layer_norm_eps, profiler)
    policy = make_policy(args.gpu_batch_size, args.batch_size // args.gpu_batch_size, not args.no_overlap)
    comparison_error = None
    with tempfile.TemporaryDirectory(prefix="flexgen-cemu-fdm-", dir=args.fdm_dir) as directory, ExitStack() as stack:
        print("[flexgen-cemu] setup: opening KV backend and Attention slots", flush=True)
        store, backends = open_backends(stack, args, config, directory, runtime, profiler)
        print("[flexgen-cemu] backend ready; starting checkpoint validation/preload", flush=True)
        preload_started = time.perf_counter()
        with profiler.measure("setup.checkpoint_preload_wall"):
            model = create_model(config, env, args.weights_dir, policy, backends, profiler)
            model.sync()
        stack.callback(model.delete_all_weights)
        stack.callback(model.cemu_session.close)
        print(f"[flexgen-cemu] checkpoint loaded in {time.perf_counter() - preload_started:.6f}s; GPU weights ready", flush=True)
        framework.timers("generate").reset()
        print(f"[flexgen-cemu] generation begin: Prefill + {args.decode_steps} Decode steps", flush=True)
        with torch.inference_mode(), profiler.measure("pipeline.e2e_including_drain_wall"):
            with profiler.measure("pipeline.generate_wall"):
                output = model.generate(validate_inputs(model, prompt, args.decode_steps),
                                        max_new_tokens=args.decode_steps + 1)
                model.sync()
            print("[flexgen-cemu] generation complete; final KV drain begin", flush=True)
            with profiler.measure("pipeline.final_flush_wall"):
                model.cemu_session.drain()
        print("[flexgen-cemu] final KV drain complete; checking results", flush=True)
        costs = framework.timers("generate").costs
        if len(costs) != args.decode_steps + 1:
            raise RuntimeError("FlexGen did not execute the requested generation steps")
        profiler.record("framework.prefill_iteration_wall", round(costs[0] * 1e9))
        profiler.record("framework.decode_iterations_wall", round(sum(costs[1:]) * 1e9))
        generation_timer = framework.timers("generate")
        if profiler.interval_path:
            if len(generation_timer.start_times) != len(costs) or len(generation_timer.stop_times) != len(costs):
                raise RuntimeError("missing real FlexGen iteration timestamps for stack accounting")
            for index, (start, end) in enumerate(zip(generation_timer.start_times, generation_timer.stop_times)):
                profiler.trace_interval("framework.prefill_iteration_wall" if index == 0 else
                                        "framework.decode_iterations_wall", round(start * 1e9), round(end * 1e9))
        stats = store.statistics()
        if (stats["failed"] or not stats["sealed"] or
                model.cemu_session.submitted != args.decode_steps * config.num_hidden_layers * policy.num_gpu_batches or
                stats["client_appended_head_tokens"] != (args.prompt_length + args.decode_steps) *
                    args.batch_size * config.n_head * config.num_hidden_layers or
                any(stats[key] != args.prompt_length + args.decode_steps
                    for key in ("accepted_stream0", "token_k", "channel_k", "token_v"))):
            raise RuntimeError(f"incomplete KV drain: {stats}")
        result = dict(framework=str(framework.__file__), storage=args.storage, clocksource=clock,
            arguments=vars(args), statistics=stats, tokens=output[:, args.prompt_length:].tolist(),
            attention_submissions=model.cemu_session.submitted, peak_pending=model.cemu_session.peak_pending,
            prefill_seconds=costs[0], decode_step_seconds=list(costs[1:]),
            case=case, prompt_file=str(prompt_output), token_comparison="not_requested")
        if reference is not None:
            try:
                compare_tokens(result["tokens"], reference)
                result["token_comparison"] = "pass"
            except ValueError as error:
                comparison_error = error
                result["token_comparison"] = "fail"
                result["comparison_error"] = str(error)
        if tokenizer is not None:
            result["text"] = decode_output(tokenizer, prompt, result["tokens"])
            for batch, (continuation, full) in enumerate(zip(result["text"]["continuation"], result["text"]["full"])):
                print(f"[flexgen-cemu] text batch={batch}, continuation={continuation!r}, full={full!r}", flush=True)
        write_json(args.result_output, result)
        verdict = "FAIL" if comparison_error else "PASS"
        print(f"[flexgen-cemu] {verdict}: OptLM.generate -> {args.storage} KV -> autonomous SparF; tokens={result['tokens']}", flush=True)
        print(f"[flexgen-cemu] token comparison={result['token_comparison']}; full-model logits not checked", flush=True)
        print(f"[flexgen-cemu] writeback statistics={stats}", flush=True)
    profiler.write_csv(args.profile_output)
    if args.profile_stack:
        print("[flexgen-profile] stack profiling: Guest elapsed intervals include blocking/queueing; "
              "client io_wait includes device service. Nested/worker intervals overlap: do not sum them. "
              "CUDA stream time and Host device timing are separate clock domains.")
    for line in profiler.summary():
        print(f"[flexgen-profile] {line}")
    if comparison_error is not None:
        raise comparison_error


if __name__ == "__main__":
    main()
