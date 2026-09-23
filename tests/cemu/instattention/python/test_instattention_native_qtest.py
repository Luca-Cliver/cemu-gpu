import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import unittest

from native_kv_client import (
    BUFFER_APPEND, BUFFER_REGISTER, BUFFER_STATUS, INFO, READ, VERSION,
    buffer_row, checked, coordinates, run_buffer_contract, run_contract,
)


ROOT = Path(__file__).resolve().parents[4]
QEMU = Path(os.environ.get("CEMU_QEMU_BINARY", ROOT / "build/qemu-system-x86_64"))
CONFIG = ROOT / "tests/cemu/instattention/experiments/configs/native_kv_smoke.json"
BAR = 0xF0000000
ASQ, ACQ, IOSQ, IOCQ, DBS, EIS, DATA = [0x100000 + index * 0x10000 for index in range(7)]


class NativeQtest:
    def __init__(self, directory, config=CONFIG):
        self.directory = Path(directory)
        self.backend = self.directory / "new-nvm.img"
        self.log = open(self.directory / "qemu.log", "w+")
        self.process = subprocess.Popen([
            str(QEMU), "-machine", "pc", "-accel", "qtest", "-m", "64M",
            "-display", "none", "-nodefaults", "-qtest-log", "/dev/null",
            "-qtest", f"unix:{directory}/qtest.sock,server=on,wait=off",
            "-qmp", f"unix:{directory}/qmp.sock,server=on,wait=off",
            "-device", f"femu,id=native0,addr=04.0,ftl_mode=instattention,config_file={config},"
                       f"nvm_backend_file={self.backend}",
        ], cwd=directory, stdout=self.log, stderr=self.log,
            env={**os.environ, "CEMU_CUDA_SYNC": "0"})
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(5)
        self.stream = None
        self.qmp = None
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    self.socket.connect(f"{directory}/qtest.sock")
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if self.process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(self.read_log())
                    time.sleep(0.01)
            self.stream = self.socket.makefile("rwb", buffering=0)
            self.qmp = socket.socket(socket.AF_UNIX)
            self.qmp.settimeout(5)
            self.qmp.connect(f"{directory}/qmp.sock")
            self.qmp_stream = self.qmp.makefile("rwb", buffering=0)
            self.qmp_stream.readline()
            self.qmp_command("qmp_capabilities")
            self.command("outl 0xcf8 0x80002010")
            self.command(f"outl 0xcfc {BAR:#x}")
            self.command("outl 0xcf8 0x80002004")
            self.command("outw 0xcfc 6")
            self.initialize()
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.socket.close()
            if self.stream:
                self.stream.close()
            if self.qmp:
                self.qmp.close()
            self.log.close()
            raise

    def read_log(self):
        self.log.flush()
        return (self.directory / "qemu.log").read_text()

    def command(self, command):
        self.socket.sendall((command + "\n").encode())
        while True:
            line = self.stream.readline().decode().strip()
            if line.startswith("IRQ"):
                continue
            if not line.startswith("OK"):
                raise RuntimeError(f"qtest {command[:80]}: {line}; {self.read_log()}")
            return line.split(maxsplit=1)[1] if " " in line else ""

    def qmp_command(self, name, arguments=None):
        command = {"execute": name}
        if arguments is not None:
            command["arguments"] = arguments
        self.qmp.sendall((json.dumps(command) + "\n").encode())
        while True:
            result = json.loads(self.qmp_stream.readline())
            if "return" in result or "error" in result:
                if "error" in result:
                    raise RuntimeError(result)
                return result

    def write(self, address, data):
        self.command(f"write {address:#x} {len(data)} 0x{data.hex()}")

    def read(self, address, size):
        return bytes.fromhex(self.command(f"read {address:#x} {size}").removeprefix("0x"))

    def initialize(self):
        self.command(f"writel {BAR + 0x14:#x} 0")
        for address in (ASQ, ACQ, IOSQ, IOCQ, DBS, EIS, DATA):
            self.write(address, bytes(4096))
        self.tails = [0, 0]
        self.heads = [0, 0]
        self.phases = [1, 1]
        self.cid = 0
        self.command(f"writel {BAR + 0x24:#x} 0x000f000f")
        self.command(f"writeq {BAR + 0x28:#x} {ASQ:#x}")
        self.command(f"writeq {BAR + 0x30:#x} {ACQ:#x}")
        self.command(f"writel {BAR + 0x14:#x} 0x00460001")
        ready = int(self.command(f"readl {BAR + 0x1c:#x}"), 0)
        if ready != 1:
            raise RuntimeError(f"controller not ready: {ready}; {self.read_log()}")
        for opcode, address, fields in (
            (5, IOCQ, (0x000F0001, 1, 0, 0, 0, 0)),
            (1, IOSQ, (0x000F0001, 0x10001, 0, 0, 0, 0)),
            (0x7C, DBS, (0, 0, 0, 0, 0, 0)),
        ):
            self.submit(0, opcode, fields, address, EIS if opcode == 0x7C else 0, namespace=0)
            status, _ = self.wait(0)
            if status:
                raise RuntimeError(f"admin opcode={opcode:#x} status={status:#x}; {self.read_log()}")

    def submit(self, queue, opcode, fields, prp1=DATA, prp2=0, namespace=1, command2=0, command4=0):
        self.cid = (self.cid + 1) % 65536
        command = struct.pack("<BBHIQQQQ6I", opcode, 0, self.cid, namespace,
                              command2, command4, prp1, prp2, *fields)
        self.write((ASQ if queue == 0 else IOSQ) + self.tails[queue] * 64, command)
        self.tails[queue] = (self.tails[queue] + 1) % 16
        doorbell = BAR + 0x1000 if queue == 0 else DBS + 8
        self.command(f"writel {doorbell:#x} {self.tails[queue]}")

    def completion(self, queue):
        data = self.read((ACQ if queue == 0 else IOCQ) + self.heads[queue] * 16, 16)
        result, _, _, _, _, status = struct.unpack("<IIHHHH", data)
        return (status >> 1, result) if status & 1 == self.phases[queue] else None

    def wait(self, queue):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.completion(queue)
            if result is not None:
                self.heads[queue] = (self.heads[queue] + 1) % 16
                if self.heads[queue] == 0:
                    self.phases[queue] ^= 1
                doorbell = BAR + 0x1004 if queue == 0 else DBS + 12
                self.command(f"writel {doorbell:#x} {self.heads[queue]}")
                return result
            self.command("clock_step 100000")
            time.sleep(0.001)
        raise AssertionError(f"completion timeout queue={queue}; {self.read_log()}")

    def __call__(self, opcode, fields, payload=None, namespace=1):
        self.write(DATA, payload if payload is not None else bytes(4096))
        self.submit(1, opcode, fields, namespace=namespace)
        status, result = self.wait(1)
        return status, result, self.read(DATA, 4096)

    def close(self):
        try:
            self.qmp_command("qom-set", {"path": "/machine/peripheral/native0",
                                       "property": "realized", "value": False})
            self.qmp_command("quit")
            self.process.wait(timeout=5)
            if self.process.returncode != 0:
                raise AssertionError(self.read_log())
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            self.stream.close()
            self.socket.close()
            self.qmp_stream.close()
            self.qmp.close()
            self.log.close()


@unittest.skipUnless(QEMU.exists(), "build qemu-system-x86_64 first")
class NativeDeviceTest(unittest.TestCase):
    def test_native_buffer(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-buffer-") as directory:
            device = NativeQtest(directory)
            try:
                run_buffer_contract(device)
            finally:
                device.close()

    def test_buffer_exit_with_frozen_deadline(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-buffer-stop-") as directory:
            config = json.loads(CONFIG.read_text())
            config["namespaces"][0]["parameters"]["pg_wr_lat"] = 2000000000
            path = Path(directory) / "long-write.json"
            path.write_text(json.dumps(config))
            device = NativeQtest(directory, path)
            try:
                packet = struct.pack("<8IQI", 1, 1, 1, 128, 64, 8, 2, 4096, 1024 * 1024, 32)
                checked(device, BUFFER_REGISTER, coordinates(1), packet.ljust(4096, b"\0"))
                for first in (0, 8):
                    payload = b"".join(buffer_row(0, token) for token in range(first, first + 8))
                    payload += b"".join(buffer_row(0, token, True) for token in range(first, first + 8))
                    checked(device, BUFFER_APPEND, coordinates(1, token=first, valid=8), payload)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    stats = struct.unpack_from("<11Q", checked(device, BUFFER_STATUS, coordinates(1)))
                    if stats[8]:
                        break
                    time.sleep(0.001)
                else:
                    self.fail("background write did not start")
                self.assertEqual(stats[1:4], (0, 0, 0))
                self.assertGreater(stats[8], 2000000000)
            finally:
                device.close()

    def test_startup_guards(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-guards-") as directory:
            directory = Path(directory)
            backend = directory / "existing.img"
            sentinel = b"existing experiment must not be touched"
            backend.write_bytes(sentinel)
            invalid = directory / "invalid.json"
            configuration = json.loads(CONFIG.read_text())
            configuration["namespaces"][0]["parameters"]["nchs"] = 0
            invalid.write_text(json.dumps(configuration))
            for mode, config, path in (
                ("instattention", CONFIG, backend),
                ("unknown", CONFIG, directory / "unknown.img"),
                ("instattention", invalid, directory / "invalid.img"),
            ):
                with self.subTest(mode=mode, config=config):
                    result = subprocess.run([
                        str(QEMU), "-machine", "pc", "-accel", "qtest", "-display", "none",
                        "-nodefaults", "-m", "64M", "-device",
                        f"femu,ftl_mode={mode},config_file={config},nvm_backend_file={path}",
                    ], cwd=directory, env={**os.environ, "CEMU_CUDA_SYNC": "0"},
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
                    self.assertGreater(result.returncode, 0, result.stdout.decode())
                    self.assertEqual(backend.read_bytes(), sentinel)
                    if path != backend:
                        self.assertFalse(path.exists())

    def test_completion_backpressure(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-cq-") as directory:
            device = NativeQtest(directory)
            try:
                fields = (VERSION, 0, 0, 0, 0, 0)
                for _ in range(15):
                    device.submit(1, INFO, fields)
                for _ in range(100):
                    device.command("clock_step 100000")
                    time.sleep(0.001)
                    if struct.unpack_from("<H", device.read(IOCQ + 14 * 16, 16), 14)[0] & 1:
                        break
                else:
                    self.fail("CQ did not fill")
                first = device.read(IOCQ, 16)
                for _ in range(2):
                    device.submit(1, INFO, fields)
                device.command("clock_step 1000000")
                time.sleep(0.05)
                self.assertEqual(device.read(IOCQ, 16), first, "full CQ overwritten")
                device.submit(0, 0, (1, 0, 0, 0, 0, 0), namespace=0)
                self.assertNotEqual(device.wait(0)[0], 0, "deleted SQ with pending IO")
                for _ in range(17):
                    self.assertEqual(device.wait(1)[0], 0)
                device.submit(1, INFO, fields)
                time.sleep(0.05)
                self.assertIsNone(device.completion(1))
            finally:
                device.close()

    def test_full_cq_does_not_block_another_queue(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-multicq-") as directory:
            device = NativeQtest(directory)
            try:
                sq2, cq2, data2 = 0x180000, 0x190000, 0x1A0000
                for address in (sq2, cq2, data2):
                    device.write(address, bytes(4096))
                for opcode, address, fields in (
                    (5, cq2, (0x000F0002, 1, 0, 0, 0, 0)),
                    (1, sq2, (0x000F0002, 0x20001, 0, 0, 0, 0)),
                ):
                    device.submit(0, opcode, fields, address, namespace=0)
                    self.assertEqual(device.wait(0)[0], 0)
                fields = (VERSION, 0, 0, 0, 0, 0)
                for _ in range(15):
                    device.submit(1, INFO, fields)
                for _ in range(200):
                    device.command("clock_step 100000")
                    time.sleep(0.001)
                    if struct.unpack_from("<H", device.read(IOCQ + 14 * 16, 16), 14)[0] & 1:
                        break
                else:
                    self.fail("CQ1 did not fill")
                first = device.read(IOCQ, 16)
                device.submit(1, INFO, fields)
                device.command("clock_step 1000000")
                time.sleep(0.05)
                device.write(sq2, struct.pack("<BBHIQQQQ6I", INFO, 0, 999, 1,
                                              0, 0, data2, 0, *fields))
                device.command(f"writel {DBS + 16:#x} 1")
                time.sleep(0.02)
                self.assertEqual(device.read(cq2, 16), bytes(16), "ignored target deadline")
                for _ in range(200):
                    device.command("clock_step 100000")
                    time.sleep(0.001)
                    status = struct.unpack_from("<H", device.read(cq2, 16), 14)[0]
                    if status & 1:
                        break
                else:
                    self.fail("full CQ1 blocked CQ2")
                self.assertEqual(status, 1)
                self.assertEqual(struct.unpack_from("<H", device.read(cq2, 16), 12)[0], 999)
                self.assertEqual(device.read(IOCQ, 16), first, "full CQ1 overwritten")
                for _ in range(16):
                    self.assertEqual(device.wait(1)[0], 0)
                device.submit(1, INFO, fields)
                self.assertEqual(device.wait(1)[0], 0)
            finally:
                device.close()

    def test_protocol_and_reset(self):
        with tempfile.TemporaryDirectory(prefix="native-kv-qtest-") as directory:
            device = NativeQtest(directory)
            try:
                task = run_contract(device)
                device.submit(0, 4, (1, 0, 0, 0, 0, 0), namespace=0)
                self.assertNotEqual(device.wait(0)[0], 0, "deleted CQ still referenced by SQ")
                status, _, _ = device(INFO, (VERSION, 0, 0, 0, 0, 0), namespace=2)
                self.assertNotEqual(status, 0)
                device.submit(1, READ, coordinates(task))
                time.sleep(0.05)
                self.assertIsNone(device.completion(1), "CQ completed without advancing target time")
                device.initialize()
                actual = checked(device, READ, coordinates(task))
                self.assertEqual(actual, bytes(offset * 13 % 256 for offset in range(4096)))
                for opcode in (2, 8, 9):
                    checked(device, opcode, (0, 0, 0, 0, 0, 0), rejected=True)
                device.submit(0, 0x80, (0, 0, 0, 0, 0, 0))
                self.assertNotEqual(device.wait(0)[0], 0, "FORMAT must not reset native mapping")
                self.assertIn("mode=instattention", device.read_log())
            finally:
                device.close()


if __name__ == "__main__":
    unittest.main()
