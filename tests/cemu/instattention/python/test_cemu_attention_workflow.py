#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "build-guest"))
sys.path.insert(0, str(PROJECT_DIRECTORY / "python"))

from _cemu_client import get_file_extents
from cemu_flexgen.attention_workflow_abi import (
    ATTENTION_WORKFLOW_COMMAND,
    ATTENTION_WORKFLOW_TRACE_BYTES,
    AttentionWorkflowTrace,
    pack_attention_workflow,
)
from cemu_flexgen.cemu_device import CemuDevice, RangeSpec
from test_attention_phases import make_case, padded_tokens, reference
from test_cemu_attention_phases import require_fdmfs


def aligned_range(directory, name, size):
    return RangeSpec(str(Path(directory) / name), (size + 511) // 512 * 512)


def write_nvm_file(path, value):
    contiguous = np.ascontiguousarray(value)
    with open(path, "wb", buffering=0) as output:
        written = output.write(contiguous)
        if written != contiguous.nbytes:
            raise OSError(
                f"short NVM write for {path}: wrote {written} of {contiguous.nbytes} bytes"
            )
        os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description="Test one-request CEMU Attention K/QK-V/PV workflow")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--qk-runtime-ns", type=int, default=200000)
    parser.add_argument("--pv-runtime-ns", type=int, default=100000)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    require_fdmfs(args.fdm_dir)
    if not os.path.isdir(args.nvm_dir):
        raise FileNotFoundError(f"NVM directory does not exist: {args.nvm_dir}")

    config, query, keys, values = make_case(
        args.dtype, batch=1, query_heads=40, kv_heads=40,
        dimension=128, tokens=args.tokens,
    )
    key_storage = padded_tokens(config, keys)
    value_storage = padded_tokens(config, values)
    expected_probability, expected_output = reference(config, query, keys, values)
    program = args.program or (
        "./build/attention_phases_devptr.so" if args.cuda
        else "./build/attention_phases.so"
    )

    with tempfile.TemporaryDirectory(prefix="cemu-workflow-nvm-", dir=args.nvm_dir) as nvm_directory:
        key_path = Path(nvm_directory) / "keys"
        value_path = Path(nvm_directory) / "values"
        write_nvm_file(key_path, key_storage)
        write_nvm_file(value_path, value_storage)
        key_extents = get_file_extents(str(key_path), config.kv_bytes)
        value_extents = get_file_extents(str(value_path), config.kv_bytes)
        with tempfile.TemporaryDirectory(prefix="cemu-workflow-fdm-", dir=args.fdm_dir) as fdm_directory:
            ranges = [
                aligned_range(fdm_directory, "query", config.query_bytes),
                aligned_range(fdm_directory, "key_staging", config.kv_bytes),
                aligned_range(fdm_directory, "value_staging", config.kv_bytes),
                aligned_range(fdm_directory, "probability", config.probability_bytes),
                aligned_range(fdm_directory, "output", config.query_bytes),
                aligned_range(fdm_directory, "trace", ATTENTION_WORKFLOW_TRACE_BYTES),
            ]
            with CemuDevice(
                program_name=f"attention_workflow_{Path(fdm_directory).name}",
                program_path=program,
                function_name="attention_phase",
                ranges=ranges,
                control_path=args.control,
                namespace_path=args.namespace,
                cuda_target=args.cuda,
            ) as device:
                device.write_tensor(0, query)
                for serial in (False, True):
                    metadata = pack_attention_workflow(
                        config, key_extents, value_extents,
                        qk_runtime_ns=args.qk_runtime_ns,
                        pv_runtime_ns=args.pv_runtime_ns,
                        serial=serial,
                    )
                    mode = "serial" if serial else "overlap"
                    print(f"[attention-workflow] submit one execute mode={mode}, extents=({len(key_extents)},{len(value_extents)})", flush=True)
                    result = device.execute(
                        cparam1=ATTENTION_WORKFLOW_COMMAND, metadata=metadata,
                    )
                    if result != query.size:
                        raise AssertionError(
                            f"device workflow returned {result}, expected {query.size} output elements"
                        )
                    probability = device.read_tensor(
                        3, expected_probability.shape, np.float32,
                    )
                    output = device.read_tensor(4, query.shape, config.dtype)
                    trace_bytes = device.read_tensor(
                        5, (ATTENTION_WORKFLOW_TRACE_BYTES,), np.uint8,
                    )
                    trace = AttentionWorkflowTrace.unpack(trace_bytes)
                    np.testing.assert_allclose(probability, expected_probability, rtol=2e-5, atol=2e-6)
                    tolerance = 1e-3 if config.dtype == np.float16 else 2e-5
                    np.testing.assert_allclose(output, expected_output, rtol=tolerance, atol=tolerance)
                    expected_model = (
                        trace.key_model_ns +
                        ((trace.qk_model_ns + trace.value_model_ns) if serial else
                         max(trace.qk_model_ns, trace.value_model_ns)) +
                        trace.pv_model_ns
                    )
                    if trace.total_model_ns != expected_model:
                        raise AssertionError("device workflow critical-path model is inconsistent")
                    if serial:
                        if trace.value_submit_ns < trace.qk_done_ns:
                            raise AssertionError("serial V read started before QK completed")
                    else:
                        if trace.value_submit_ns > trace.qk_start_ns:
                            raise AssertionError("V read was not submitted before QK")
                        if trace.qk_value_actual_overlap_ns == 0:
                            raise AssertionError("QK and V read did not overlap in real execution")
                    if trace.pv_start_ns < max(trace.qk_done_ns, trace.value_ready_ns):
                        raise AssertionError("PV started before QK and V were both ready")
                    print(
                        f"[attention-workflow] mode={mode} actual: "
                        f"K={(trace.key_ready_ns-trace.key_submit_ns)/1000:.3f}us, "
                        f"QK={(trace.qk_done_ns-trace.qk_start_ns)/1000:.3f}us, "
                        f"V={(trace.value_ready_ns-trace.value_submit_ns)/1000:.3f}us, "
                        f"overlap={trace.qk_value_actual_overlap_ns/1000:.3f}us, "
                        f"PV={(trace.pv_done_ns-trace.pv_start_ns)/1000:.3f}us; "
                        f"modeled={trace.total_model_ns}ns",
                        flush=True,
                    )
    print(f"[attention-workflow] PASS target={'cuda' if args.cuda else 'cpu'}; one Guest execute per logical Attention")


if __name__ == "__main__":
    main()
