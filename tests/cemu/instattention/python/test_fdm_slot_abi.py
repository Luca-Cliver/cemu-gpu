from pathlib import Path
import subprocess
import tempfile
import unittest


class FdmSlotAbiTest(unittest.TestCase):
    def test_guest_and_kernel_ioctl_sizes_and_numbers_match(self):
        root = Path(__file__).resolve().parents[4]
        headers = [root / "tests/cemu/cemu_ioctl.h",
                   root.parent / "linux-cemu/include/uapi/linux/cemu_ioctl.h"]
        if not all(path.is_file() for path in headers):
            self.skipTest("requires CEMU and linux-cemu source trees")
        outputs = []
        with tempfile.TemporaryDirectory(prefix="fdm-slot-abi-") as directory:
            source = Path(directory) / "check.c"
            binary = Path(directory) / "check"
            for header in headers:
                source.write_text(
                    f'#include "{header}"\n#include <stddef.h>\n#include <stdio.h>\n'
                    'int main(void) {\n'
                    ' printf("%zu %zu %zu %lu %lu %lu\\n",\n'
                    ' sizeof(struct ioctl_create_mrs), sizeof(struct ioctl_fdm_kv_bind),\n'
                    ' offsetof(struct ioctl_fdm_kv_bind, addr),\n'
                    ' (unsigned long)IOCTL_CEMU_CREATE_MRS_PINNED,\n'
                    ' (unsigned long)IOCTL_CEMU_BIND_FDM_KV,\n'
                    ' (unsigned long)IOCTL_CEMU_DELETE_MRS);\n return 0;\n}\n')
                subprocess.run(["cc", "-Wall", "-Werror", str(source), "-o", str(binary)], check=True)
                outputs.append(subprocess.check_output([str(binary)], text=True))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0].split()[:3], ["40", "16", "8"])


if __name__ == "__main__":
    unittest.main()
