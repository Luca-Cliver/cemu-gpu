#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from experiments.config import load_experiment_config
from experiments.runtime_model import SparfAttentionRuntimeModel
from experiments.softmax_ratio import IMPLEMENTATION, SoftmaxRatioTable, sparf_softmax_shapes


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CUDA Softmax shape ratios, not FPGA timing")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/opt13b_sparf_1csd.json")
    parser.add_argument("--device", help="explicit CUDA device, e.g. cuda:1; never chosen automatically")
    parser.add_argument("--batch-sizes", type=positive_integer, nargs="+", default=[4],
                        help="batch sizes inside one CSD request (GPU microbatch sizes)")
    parser.add_argument("--prompt-length", type=positive_integer)
    parser.add_argument("--decode-steps", type=positive_integer, default=16)
    parser.add_argument("--compression-ratio", type=positive_integer)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--iterations", type=positive_integer, default=200)
    parser.add_argument("--repeats", type=positive_integer, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="print shapes without importing torch or touching GPUs")
    return parser.parse_args()


def summarize(vectors, tokens, samples):
    return dict(vectors=vectors, tokens=tokens, median_us=statistics.median(samples),
                min_us=min(samples), max_us=max(samples), samples_us=samples)


def measure(torch, device, vectors, tokens, args):
    with torch.cuda.device(device), torch.inference_mode():
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            generator = torch.Generator(device=device).manual_seed(0)
            scores = torch.randn((vectors, tokens), device=device, dtype=torch.float32, generator=generator)
            output = torch.empty_like(scores)
            for _ in range(args.warmup):
                torch.softmax(scores, dim=-1, out=output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.iterations):
                torch.softmax(scores, dim=-1, out=output)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        with torch.cuda.stream(stream):
            graph.replay()
            stream.synchronize()
            for _ in range(args.repeats):
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / args.iterations)
            reference = torch.softmax(scores.double(), dim=-1).float()
            torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-7)
        stream.synchronize()
    return summarize(vectors, tokens, samples)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    prompt = args.prompt_length or config.workload.prompt_length
    compression = args.compression_ratio or config.workload.compression_ratio
    if prompt + args.decode_steps > config.model.max_sequence_length:
        raise ValueError("prompt + Decode steps exceeds model capacity")
    anchor_shape = (config.instcsd.softmax_anchor_heads, config.instcsd.softmax_anchor_tokens)
    shapes = sparf_softmax_shapes(args.batch_sizes, config.model.num_query_heads,
                                  prompt, args.decode_steps, compression)
    shapes = sorted(set(shapes) - {anchor_shape})
    print(f"[softmax-ratio] anchor={anchor_shape}, target_shapes={len(shapes)}, "
          f"prompt={prompt}, decode_steps={args.decode_steps}, microbatches={args.batch_sizes}")
    if args.dry_run:
        print(f"[softmax-ratio] target shapes={shapes}; no CUDA work performed")
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite measurement: {args.output}")
    if not args.device:
        raise ValueError("--device cuda:N is required; select an idle GPU explicitly")
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must name an explicit CUDA index, e.g. cuda:1")
    properties = torch.cuda.get_device_properties(device)
    print(f"[softmax-ratio] device={device}, name={properties.name}, implementation={IMPLEMENTATION}")
    print("[softmax-ratio] isolated FP32 score Softmax; no checkpoint, CEMU, QK, Top-k, I/O or copies in timing")
    before = measure(torch, device, *anchor_shape, args)
    measurements = []
    for index, (vectors, tokens) in enumerate(shapes):
        result = measure(torch, device, vectors, tokens, args)
        measurements.append(result)
        if index < 2 or (index + 1) % 50 == 0 or index + 1 == len(shapes):
            print(f"[softmax-ratio] {index + 1}/{len(shapes)} shape=({vectors},{tokens}) "
                  f"median={result['median_us']:.6f}us", flush=True)
    after = measure(torch, device, *anchor_shape, args)
    drift = abs(after["median_us"] / before["median_us"] - 1)
    print(f"[softmax-ratio] anchor before={before['median_us']:.6f}us, "
          f"after={after['median_us']:.6f}us, drift={drift:.1%}")
    if drift > 0.2:
        raise RuntimeError("anchor drift exceeds 20%; no report written; rerun on an idle GPU")
    anchor = summarize(*anchor_shape, before["samples_us"] + after["samples_us"])
    measurements.insert(0, anchor)
    for result in measurements:
        result["ratio"] = result["median_us"] / anchor["median_us"]
        result["target_softmax_us"] = config.instcsd.softmax_anchor_latency_us * result["ratio"]
    report = dict(
        schema_version=1, implementation=IMPLEMENTATION,
        scope="Host CUDA shape ratios anchored to paper latency; not validated FPGA scaling",
        created_utc=datetime.now(timezone.utc).isoformat(),
        device=dict(index=device.index, name=properties.name, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    multiprocessors=properties.multi_processor_count),
        timing=dict(method="CUDA events around repeated CUDA graph nodes; microseconds per Softmax",
                    iterations=args.iterations, repeats=args.repeats, warmup=args.warmup,
                    anchor_before_us=before["median_us"], anchor_after_us=after["median_us"], anchor_drift=drift),
        workload=dict(batch_sizes=args.batch_sizes, heads=config.model.num_query_heads,
                      prompt_length=prompt, decode_steps=args.decode_steps, compression_ratio=compression),
        paper_anchor_latency_us=config.instcsd.softmax_anchor_latency_us,
        anchor=anchor, measurements=measurements,
    )
    table = SoftmaxRatioTable(report)
    linear = SparfAttentionRuntimeModel(config.instcsd)
    empirical = SparfAttentionRuntimeModel(config.instcsd, softmax_ratios=table)
    for batch in args.batch_sizes:
        estimates = []
        for model in (linear, empirical):
            total_ns = 0
            for tokens in range(prompt + 1, prompt + args.decode_steps + 1):
                selected = (tokens + compression - 1) // compression
                result = model.estimate(batch, config.model.num_query_heads, config.model.head_dim,
                                        tokens, config.workload.top_r, selected)
                total_ns += result.approximate_ns + result.exact_qk_ns + result.pv_ns
            estimates.append(total_ns * config.model.num_layers / 1e9)
        print(f"[softmax-ratio] microbatch={batch}, compute-only sum over {config.model.num_layers} layers "
              f"and {args.decode_steps} steps: linear={estimates[0]:.6f}s, ratio={estimates[1]:.6f}s; "
              "not end-to-end latency")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    print(f"[softmax-ratio] report={args.output}; FPGA scaling remains unvalidated")


if __name__ == "__main__":
    main()
