import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
BUILD = Path(os.environ.get("CEMU_QEMU_BUILD_DIR", ROOT / "build")).resolve()


class InstattentionQemuIoTest(unittest.TestCase):
    def test_native_internal_entry(self):
        self.run_sources([
            "hw/femu/bbssd/ftl.c",
            "hw/femu/backend/backend.c",
            "hw/femu/bbssd/instattention_mapping.c",
            "hw/femu/bbssd/instattention_io.c",
            "hw/femu/bbssd/instattention_device.c",
            "hw/femu/bbssd/instattention_buffer.c",
            "tests/cemu/instattention/guest/test_instattention_device_io.c",
        ])

    def test_production_adapters(self):
        self.run_sources([
            "hw/femu/bbssd/ftl.c",
            "hw/femu/backend/backend.c",
            "hw/femu/bbssd/instattention_mapping.c",
            "hw/femu/bbssd/instattention_io.c",
            "tests/cemu/instattention/guest/test_instattention_qemu_io.c",
        ])

    def test_cuda_copy_errors(self):
        self.run_sources([
            "hw/femu/lib/pqueue.c",
            "tests/cemu/instattention/guest/test_cuda_copy_errors.c",
        ])

    def run_sources(self, sources):
        with tempfile.TemporaryDirectory(prefix="cemu-native-qemu-") as directory:
            executable = self.build_sources(sources, directory)
            subprocess.run([str(executable)], check=True, timeout=30)

    def build_sources(self, sources, directory, shared=False):
        database = BUILD / "compile_commands.json"
        if not database.exists():
            self.skipTest("configure/build QEMU first, or set CEMU_QEMU_BUILD_DIR")
        rows = json.loads(database.read_text())
        row = next(item for item in rows if item["file"].endswith("hw/femu/bbssd/ftl.c"))
        arguments = row.get("arguments") or shlex.split(row["command"])
        flags = []
        position = 1
        while position < len(arguments):
            argument = arguments[position]
            if argument in ("-o", "-MF", "-MQ", "-MT"):
                position += 2
                continue
            if argument not in ("-c", "-MD", "-MMD", row["file"]):
                flags.append(argument)
            position += 1
        objects = []
        for index, source in enumerate(sources):
            output = Path(directory) / f"source-{index}.o"
            subprocess.run([
                arguments[0], *flags, "-ffunction-sections", "-fdata-sections",
                *(["-fPIC", "-fvisibility=hidden" if source.startswith("hw/") else
                   "-fvisibility=default", "-Wno-missing-prototypes"] if shared else []),
                "-c", str(ROOT / source), "-o", str(output),
            ], cwd=row["directory"], check=True, timeout=120)
            objects.append(str(output))
        libraries = shlex.split(subprocess.check_output([
            "pkg-config", "--libs", "glib-2.0", "gthread-2.0",
        ], text=True))
        executable = Path(directory) / ("native.so" if shared else "qemu-io-test")
        subprocess.run([
            arguments[0], "-Wl,--gc-sections", *(["-shared"] if shared else []), *objects,
            *libraries, "-ldl", "-pthread", "-lm",
            "-o", str(executable),
        ], check=True, timeout=120)
        return executable


if __name__ == "__main__":
    unittest.main()
