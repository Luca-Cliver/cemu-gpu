#!/usr/bin/env python3

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


INSTATTENTION_DIR = Path(__file__).resolve().parents[1]
TEST_CEMU_DIR = INSTATTENTION_DIR.parent
sys.path.insert(0, str(INSTATTENTION_DIR))

from experiments import load_experiment_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "opt13b_dense_1csd.json"
PIPELINE = INSTATTENTION_DIR / "python" / "test_opt_cemu_pipeline.py"
GIB = 1024**3


@dataclass(frozen=True)
class WorkloadCapacity:
    batch_size: int
    gpu_batch_size: int
    gpu_batches: int
    sequence_capacity: int
    nvm_bytes: int
    fdm_bytes: int
    decode_requests: int
    decode_staging_bytes: int


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def dtype_bytes(dtype_name):
    sizes = {"float16": 2, "float32": 4}
    try:
        return sizes[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported experiment dtype: {dtype_name}") from error


def estimate_capacity(
    experiment,
    batch_size,
    gpu_batch_size,
    output_length,
    attention_slots,
):
    if batch_size % gpu_batch_size != 0:
        raise ValueError(
            f"batch size {batch_size} is not divisible by GPU batch size "
            f"{gpu_batch_size}"
        )
    if output_length < 2:
        raise ValueError("output length must include at least two generated tokens")

    model = experiment.model
    prompt_length = experiment.workload.prompt_length
    decode_steps = output_length - 1
    sequence_capacity = prompt_length + decode_steps
    if sequence_capacity > model.max_sequence_length:
        raise ValueError(
            f"prompt {prompt_length} + Decode steps {decode_steps} exceeds "
            f"model capacity {model.max_sequence_length}"
        )

    element_size = dtype_bytes(model.dtype)
    token_bytes = (
        gpu_batch_size * model.num_kv_heads * model.head_dim * element_size
    )
    token_stride = align_up(token_bytes, 512)
    layer_stride = align_up(sequence_capacity * token_stride, 4096)
    file_size = model.num_layers * layer_stride
    gpu_batches = batch_size // gpu_batch_size
    nvm_bytes = 2 * gpu_batches * file_size

    staging_bytes = sequence_capacity * token_stride
    query_bytes = align_up(
        gpu_batch_size * model.num_query_heads * model.head_dim * element_size,
        512,
    )
    state_bytes = align_up(
        gpu_batch_size * model.num_query_heads * (model.head_dim + 2) * 4,
        512,
    )
    fdm_bytes = attention_slots * (
        2 * staging_bytes + 2 * query_bytes + state_bytes
    )
    history_token_sum = (
        decode_steps * prompt_length
        + decode_steps * (decode_steps - 1) // 2
    )
    decode_requests = decode_steps * model.num_layers * gpu_batches
    decode_staging_bytes = (
        2
        * gpu_batches
        * model.num_layers
        * token_stride
        * history_token_sum
    )
    return WorkloadCapacity(
        batch_size=batch_size,
        gpu_batch_size=gpu_batch_size,
        gpu_batches=gpu_batches,
        sequence_capacity=sequence_capacity,
        nvm_bytes=nvm_bytes,
        fdm_bytes=fdm_bytes,
        decode_requests=decode_requests,
        decode_staging_bytes=decode_staging_bytes,
    )


def gibibytes(value):
    return value / GIB


def validate_checkpoint_config(model_directory, experiment):
    config_path = model_directory / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        checkpoint = json.load(config_file)
    expected = {
        "num_hidden_layers": experiment.model.num_layers,
        "hidden_size": experiment.model.hidden_size,
        "num_attention_heads": experiment.model.num_query_heads,
        "ffn_dim": experiment.model.intermediate_size,
        "max_position_embeddings": experiment.model.max_sequence_length,
    }
    mismatches = [
        f"{name}={checkpoint.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if checkpoint.get(name) != value
    ]
    if checkpoint.get("model_type") != "opt":
        mismatches.append(f"model_type={checkpoint.get('model_type')!r}, expected 'opt'")
    if mismatches:
        raise ValueError("checkpoint does not match experiment: " + "; ".join(mismatches))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the dense one-CSD OPT-13B workload shape from InstAttention "
            "Figure 14 through the functional CEMU pipeline"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument(
        "--program",
        type=Path,
        default=Path("./build/dense_attention_parallel_devptr.so"),
    )
    parser.add_argument("--csd-target", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-batch-size", type=positive_integer, default=4)
    parser.add_argument("--attention-slots", type=positive_integer, default=2)
    parser.add_argument(
        "--batch-sizes",
        type=positive_integer,
        nargs="+",
        help="subset of Figure 14 batch sizes; defaults to every configured size",
    )
    parser.add_argument(
        "--output-length",
        type=positive_integer,
        help="override the paper output length for smoke testing",
    )
    parser.add_argument(
        "--fdm-capacity-gib",
        type=nonnegative_float,
        default=2.0,
        help="configured CEMU FDM capacity used for preflight validation",
    )
    parser.add_argument(
        "--nvm-reserve-gib",
        type=nonnegative_float,
        default=1.0,
        help="free NVM space left unused by preflight validation",
    )
    parser.add_argument("--skip-capacity-check", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="write one aggregate phase-profile CSV per batch size",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = load_experiment_config(args.config)
    if experiment.model.name.lower() != "opt-13b":
        raise ValueError("Figure 14 functional runner requires the OPT-13B config")
    if experiment.instcsd.count != 1:
        raise ValueError("Figure 14 functional runner requires exactly one CSD")

    model_directory = args.model_dir.resolve()
    nvm_directory = args.nvm_dir.resolve()
    fdm_directory = args.fdm_dir.resolve()
    if not model_directory.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_directory}")
    if not nvm_directory.is_dir():
        raise FileNotFoundError(f"NVM directory does not exist: {nvm_directory}")
    if not fdm_directory.is_dir():
        raise FileNotFoundError(f"FDM directory does not exist: {fdm_directory}")
    validate_checkpoint_config(model_directory, experiment)
    profile_directory = None
    if args.profile_dir is not None:
        profile_directory = args.profile_dir.resolve()
        profile_directory.mkdir(parents=True, exist_ok=True)

    batch_sizes = tuple(args.batch_sizes or experiment.workload.batch_sizes)
    unknown = tuple(
        batch for batch in batch_sizes if batch not in experiment.workload.batch_sizes
    )
    if unknown:
        raise ValueError(
            f"batch sizes {unknown} are not part of the configured Figure 14 sweep"
        )
    output_length = args.output_length or experiment.workload.decode_length
    capacities = tuple(
        estimate_capacity(
            experiment,
            batch,
            args.gpu_batch_size,
            output_length,
            args.attention_slots,
        )
        for batch in batch_sizes
    )

    nvm_free = shutil.disk_usage(nvm_directory).free
    nvm_usable = max(0, nvm_free - int(args.nvm_reserve_gib * GIB))
    fdm_capacity = int(args.fdm_capacity_gib * GIB)
    print(
        f"[figure14-functional] model={experiment.model.name}, "
        f"prompt={experiment.workload.prompt_length}, output={output_length}, "
        f"Decode executes={output_length - 1}, batches={batch_sizes}, "
        f"gpu_batch_size={args.gpu_batch_size}, CSDs={experiment.instcsd.count}"
    )
    for capacity in capacities:
        print(
            f"[figure14-functional] batch={capacity.batch_size}: "
            f"gpu_batches={capacity.gpu_batches}, "
            f"NVM={gibibytes(capacity.nvm_bytes):.2f} GiB, "
            f"FDM={gibibytes(capacity.fdm_bytes):.2f} GiB, "
            f"CEMU_executes={capacity.decode_requests}, "
            f"Decode_KV_staging={gibibytes(capacity.decode_staging_bytes):.2f} GiB"
        )
    if not args.skip_capacity_check:
        oversized_nvm = tuple(
            item for item in capacities if item.nvm_bytes > nvm_usable
        )
        oversized_fdm = tuple(
            item for item in capacities if item.fdm_bytes > fdm_capacity
        )
        if oversized_nvm:
            required = max(item.nvm_bytes for item in oversized_nvm)
            raise RuntimeError(
                f"NVM capacity is insufficient: need up to "
                f"{gibibytes(required):.2f} GiB, usable free space is "
                f"{gibibytes(nvm_usable):.2f} GiB; select a smaller --batch-sizes "
                "subset or enlarge the CEMU NVM namespace/backend"
            )
        if oversized_fdm:
            required = max(item.fdm_bytes for item in oversized_fdm)
            raise RuntimeError(
                f"FDM capacity is insufficient: need up to "
                f"{gibibytes(required):.2f} GiB, configured capacity is "
                f"{args.fdm_capacity_gib:.2f} GiB"
            )

    for capacity in capacities:
        command = [
            sys.executable,
            str(PIPELINE),
            "--functional",
            "--model-dir",
            str(model_directory),
            "--nvm-dir",
            str(nvm_directory),
            "--fdm-dir",
            str(fdm_directory),
            "--control",
            args.control,
            "--namespace",
            args.namespace,
            "--program",
            str(args.program),
            "--csd-target",
            args.csd_target,
            "--batch-size",
            str(capacity.batch_size),
            "--gpu-batch-size",
            str(capacity.gpu_batch_size),
            "--prompt-length",
            str(experiment.workload.prompt_length),
            "--decode-steps",
            str(output_length - 1),
            "--staging-tokens",
            "0",
            "--attention-slots",
            str(args.attention_slots),
        ]
        if profile_directory is not None:
            command.extend(
                (
                    "--profile-output",
                    str(
                        profile_directory
                        / f"figure14-batch-{capacity.batch_size}-profile.csv"
                    ),
                )
            )
        print(f"[figure14-functional] run: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=TEST_CEMU_DIR, check=True)


if __name__ == "__main__":
    main()
