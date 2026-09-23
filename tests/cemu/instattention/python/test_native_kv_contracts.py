import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]


class NativeKvContractsTest(unittest.TestCase):
    def run_contract(self, name, sources, cases, extra=()):
        with tempfile.TemporaryDirectory(prefix=f"native-{name}-") as directory:
            executable = Path(directory) / "contract"
            subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c11", "-O2",
                "-Wall", "-Wextra", "-Werror", "-pedantic", "-fsanitize=undefined",
                "-fno-sanitize-recover=all", "-I" + str(ROOT), "-I" + str(ROOT / "include"),
                *(str(ROOT / source) for source in sources), *extra, "-o", str(executable),
            ], check=True, timeout=60)
            for case in cases:
                with self.subTest(contract=name, case=case):
                    subprocess.run([str(executable), *([case] if case else [])], check=True, timeout=30)

    def test_address(self):
        self.run_contract("address", ["tests/cemu/instattention/guest/test_instattention_address.c"], [None])

    def test_mapping(self):
        self.run_contract("mapping", ["hw/femu/bbssd/instattention_mapping.c",
            "tests/cemu/instattention/guest/test_instattention_mapping.c"],
            ["identity", "visibility", "data", "paper", "exhaustion", "invalid", "allocation"],
            ["-Wl,--wrap=calloc", "-Wl,--wrap=free"])

    def test_io(self):
        self.run_contract("io", ["hw/femu/bbssd/instattention_mapping.c", "hw/femu/bbssd/instattention_io.c",
            "tests/cemu/instattention/guest/test_instattention_io.c"],
            ["roundtrip", "dependencies", "failures", "validation", "bounds"])


if __name__ == "__main__":
    unittest.main()
