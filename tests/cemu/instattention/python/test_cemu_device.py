#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import CemuDevice, RangeSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Test the high-level CemuDevice wrapper")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--input", default="/mnt/fdm0/device_input")
    parser.add_argument("--output", default="/mnt/fdm0/device_output")
    parser.add_argument("--program", default="./build/vadd.so")
    return parser.parse_args()


def main():
    args = parse_args()
    element_count = 1024
    indices = np.arange(element_count, dtype=np.int32)

    input_data = np.empty(element_count * 2, dtype=np.int32)
    input_data[0::2] = indices
    input_data[1::2] = indices * 2
    output_data = np.zeros(element_count, dtype=np.int32)

    device = CemuDevice(
        program_name="device_vadd",
        program_path=args.program,
        function_name="vadd",
        ranges=[
            RangeSpec(args.input, input_data.nbytes),
            RangeSpec(args.output, output_data.nbytes),
        ],
        control_path=args.control,
        namespace_path=args.namespace,
        replace_existing=True,
    )

    assert not device.is_open
    with device:
        assert device.is_open
        assert device.program_id > 0
        assert device.memory_range_set_id > 0

        device.write_tensor(0, input_data)
        device.write_tensor(1, output_data)
        device.execute(cparam1=element_count)
        output = device.read_tensor(1, shape=(element_count,), dtype=np.int32)

    assert not device.is_open
    np.testing.assert_array_equal(output, indices * 3)
    print(f"CemuDevice vadd test passed for {element_count} elements")


if __name__ == "__main__":
    main()
