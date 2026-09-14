#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build-guest"))

from cemu_flexgen.attention_phases_abi import PV, QK_SOFTMAX
from test_attention_phases import make_case, padded_tokens, reference


def decode_mount_path(value):
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def require_fdmfs(directory):
    target = Path(directory).resolve()
    selected = None
    with open("/proc/self/mountinfo", encoding="utf-8") as mountinfo:
        for line in mountinfo:
            left, right = line.rstrip().split(" - ", 1)
            mountpoint = Path(decode_mount_path(left.split()[4]))
            try:
                target.relative_to(mountpoint)
            except ValueError:
                continue
            if selected is None or len(str(mountpoint)) > len(str(selected[0])):
                selected = (mountpoint, right.split()[0])
    if selected is None or selected[1] != "fdmfs":
        actual = "not mounted" if selected is None else f"{selected[1]} mounted at {selected[0]}"
        raise RuntimeError(
            f"--fdm-dir must reside on fdmfs: {target} ({actual}); "
            "run /root/cemu-mount.sh and verify with findmnt"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate two Attention phases with a shared FDM probability buffer")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if not os.path.isdir(args.fdm_dir):
        raise FileNotFoundError(f"FDM directory does not exist: {args.fdm_dir}")
    require_fdmfs(args.fdm_dir)
    from cemu_flexgen.cemu_device import CemuDevice, RangeSpec
    config, query, keys, values = make_case(args.dtype)
    program = args.program or ("./build/attention_phases_devptr.so" if args.cuda else "./build/attention_phases.so")
    with tempfile.TemporaryDirectory(prefix="cemu-phases-", dir=args.fdm_dir) as directory:
        def memory_range(name, size):
            return RangeSpec(str(Path(directory) / name), (size + 511) // 512 * 512)

        query_range = memory_range("query", config.query_bytes)
        key_range = memory_range("keys", config.kv_bytes)
        value_range = memory_range("values", config.kv_bytes)
        probability_range = memory_range("probabilities", config.probability_bytes)
        output_range = memory_range("output", config.query_bytes)
        common = dict(
            program_path=program, function_name="attention_phase", cuda_target=args.cuda,
            control_path=args.control, namespace_path=args.namespace,
        )
        suffix = Path(directory).name
        with CemuDevice(program_name=f"qk_{suffix}", ranges=[query_range, key_range, probability_range], **common) as qk_device:
            with CemuDevice(program_name=f"pv_{suffix}", ranges=[probability_range, value_range, output_range], **common) as pv_device:
                for iteration in range(2):
                    query *= -1
                    keys *= -1.25
                    values *= -0.5
                    expected_probabilities, expected_output = reference(config, query, keys, values)
                    qk_device.write_tensor(0, query)
                    qk_device.write_tensor(1, padded_tokens(config, keys))
                    print(f"[attention-phases] iteration={iteration} QK/Softmax: [Q,K,P], no V range", flush=True)
                    qk_device.execute(metadata=config.pack(QK_SOFTMAX))
                    pv_device.write_tensor(1, padded_tokens(config, values))
                    print("[attention-phases] V supplied after QK; PV: [P,V,output], P stays in FDM/mirror", flush=True)
                    pv_device.execute(metadata=config.pack(PV))
                    output = pv_device.read_tensor(2, query.shape, config.dtype)
                    probabilities = qk_device.read_tensor(
                        2, (config.batch_size, config.num_query_heads, config.token_count), np.float32,
                    )
                    np.testing.assert_allclose(probabilities, expected_probabilities, rtol=2e-5, atol=2e-6)
                    tolerance = 1e-3 if config.dtype == np.float16 else 2e-5
                    np.testing.assert_allclose(output, expected_output, rtol=tolerance, atol=tolerance)
        print(f"[attention-phases] PASS target={'cuda' if args.cuda else 'cpu'}, dtype={config.dtype}; two Direct executes per iteration, not device-side IO overlap")


if __name__ == "__main__":
    main()
