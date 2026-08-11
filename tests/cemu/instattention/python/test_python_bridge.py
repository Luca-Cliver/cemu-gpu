#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))

from _cemu_client import CemuClient, MemoryRange, ProgramTarget


def parse_args():
    parser = argparse.ArgumentParser(description="Test Python-to-CEMU vadd bridge")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--input", default="/mnt/fdm0/inst_py_input")
    parser.add_argument("--output", default="/mnt/fdm0/inst_py_output")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    program_path = args.program or (
        "./build/vadd_cuda_devptr.so" if args.cuda else "./build/vadd.so"
    )
    program_target = (
        ProgramTarget.CUDA_DEVICE_POINTER if args.cuda else ProgramTarget.HOST
    )
    program_name = "python_vadd_cuda" if args.cuda else "python_vadd"
    element_count = 1024
    input_data = np.empty(element_count * 2, dtype=np.int32)
    indices = np.arange(element_count, dtype=np.int32)
    input_data[0::2] = indices
    input_data[1::2] = indices * 2
    output_data = np.zeros(element_count, dtype=np.int32)

    with CemuClient(args.control, args.namespace) as client:
        client.load_program(
            program_name,
            program_path,
            "vadd",
            program_target,
            replace_existing=True,
        )
        client.activate_program()
        client.create_memory_ranges([
            MemoryRange(args.input, 0, input_data.nbytes),
            MemoryRange(args.output, 0, output_data.nbytes),
        ])
        client.write_range(0, input_data)
        client.write_range(1, output_data)
        client.execute(cparam1=element_count)
        output_bytes = client.read_range(1, output_data.nbytes)

    output = np.asarray(output_bytes).view(np.int32)
    np.testing.assert_array_equal(output, indices * 3)
    mode = "CUDA device-pointer" if args.cuda else "CPU host"
    print(
        f"Python-to-CEMU bridge passed for {element_count} vadd elements "
        f"using {mode}"
    )


if __name__ == "__main__":
    main()
