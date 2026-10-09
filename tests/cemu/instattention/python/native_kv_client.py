import argparse
import ctypes
import os
import secrets
import stat
import struct
import time


PAGE_BYTES = 4096
VERSION = 1
MAGIC = 0x31564B4E
REGISTER, INFO, WRITE, READ = 0xC1, 0xC2, 0xC5, 0xC6
BUFFER_REGISTER, BUFFER_STATUS, BUFFER_DRAIN = 0xC9, 0xCA, 0xCC
BUFFER_APPEND, BUFFER_READ = 0xCD, 0xCE
BUFFER_APPEND_BATCH = 0xD1
APPEND_MAX_BYTES = 128 * 1024


class Passthrough(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint8), ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16), ("nsid", ctypes.c_uint32),
        ("cdw2", ctypes.c_uint32), ("cdw3", ctypes.c_uint32),
        ("metadata", ctypes.c_uint64), ("addr", ctypes.c_uint64),
        ("metadata_len", ctypes.c_uint32), ("data_len", ctypes.c_uint32),
        *[(f"cdw{number}", ctypes.c_uint32) for number in range(10, 16)],
        ("timeout_ms", ctypes.c_uint32), ("result", ctypes.c_uint32),
    ]


class LinuxTransport:
    def __init__(self, path):
        if ctypes.sizeof(Passthrough) != 72:
            raise RuntimeError("unsupported NVMe ioctl ABI")
        self.descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        try:
            if not stat.S_ISCHR(os.fstat(self.descriptor).st_mode):
                raise ValueError(
                    "native KV requires a namespace character device such as /dev/ng0n1; "
                    "do not use /dev/nvme0n1: this CEMU kernel redirects block ioctls to nsid=3"
                )
            self.libc = ctypes.CDLL(None, use_errno=True)
            self.libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
            self.libc.ioctl.restype = ctypes.c_int
        except BaseException:
            os.close(self.descriptor)
            raise

    def __call__(self, opcode, fields, payload=None, namespace=1):
        size = PAGE_BYTES
        if opcode == BUFFER_APPEND_BATCH:
            if payload is None or not 0 < len(payload) <= APPEND_MAX_BYTES:
                raise ValueError("native batched APPEND requires a bounded payload")
            size = len(payload)
        data = ctypes.create_string_buffer(size)
        if payload is not None:
            if len(payload) != size:
                raise ValueError("native KV transfers exactly one page")
            data.raw = payload
        command = Passthrough(opcode=opcode, nsid=namespace,
                              addr=ctypes.addressof(data), data_len=size,
                              timeout_ms=10000)
        if opcode in (0, BUFFER_DRAIN):
            command.addr = command.data_len = 0
        for number, value in enumerate(fields, 10):
            setattr(command, f"cdw{number}", value)
        status = self.libc.ioctl(self.descriptor, 0xC0484E43, ctypes.byref(command))
        if status < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return status, command.result, data.raw

    def close(self):
        os.close(self.descriptor)


def coordinates(task, generation=1, batch=0, layer=0, head=0,
                kind=0, channel=0, token=0, valid=0):
    return (VERSION | (valid << 16), task, generation,
            batch | (layer << 16), head | (kind << 16) | (channel << 24), token)


def checked(transport, opcode, fields, payload=None, rejected=False):
    status, result, data = transport(opcode, fields, payload)
    if bool(status) != rejected:
        raise AssertionError(f"opcode={opcode:#x}: status={status:#x}, core={result}, "
                             f"expected_rejection={rejected}")
    return data


def inspect(transport):
    data = checked(transport, INFO, (VERSION, 0, 0, 0, 0, 0))
    magic, version, controller, namespace, page = struct.unpack_from("<5I", data)
    if (magic, version, page) != (MAGIC, VERSION, PAGE_BYTES):
        raise RuntimeError(f"not a native KV v1 test controller: header={data[:20].hex()}; refusing writes")
    return controller, namespace, struct.unpack_from("<4Q", data, 40)


def run_contract(transport):
    controller, namespace, before = inspect(transport)
    task = secrets.randbits(32)
    geometry = struct.pack("<8I", 2, 2, 2, 128, 259, 8, 2, PAGE_BYTES)
    checked(transport, REGISTER, coordinates(task), geometry.ljust(PAGE_BYTES, b"\0"))
    checked(transport, REGISTER, coordinates(task), geometry.ljust(PAGE_BYTES, b"\0"), rejected=True)
    cases = [
        dict(kind=0, token=0, valid=16),
        dict(kind=0, token=16, valid=16),
        dict(kind=2, token=0, valid=16),
        dict(kind=1, token=0, channel=0, valid=256),
        dict(kind=1, token=0, channel=8, valid=256),
        dict(kind=1, token=256, channel=0, valid=3),
        dict(kind=0, token=256, valid=3),
        dict(kind=0, token=0, batch=1, layer=1, head=1, valid=16),
    ]
    for index, arguments in enumerate(cases):
        payload = bytes((offset * 13 + index * 37) % 256 for offset in range(PAGE_BYTES))
        checked(transport, WRITE, coordinates(task, **arguments), payload)
        checked(transport, WRITE, coordinates(task, **arguments), payload, rejected=True)
        read_arguments = {**arguments, "valid": 0}
        for offset in (0, arguments["valid"] - 1):
            read_arguments["token"] = arguments["token"] + offset
            actual = checked(transport, READ, coordinates(task, **read_arguments))
            if actual != payload:
                raise AssertionError(f"page mismatch: {read_arguments}")
    checked(transport, READ, coordinates(task, generation=2), rejected=True)
    checked(transport, READ, coordinates(task, token=258, head=1), rejected=True)
    checked(transport, READ, coordinates(task, token=259), rejected=True)
    checked(transport, INFO, (VERSION + 1, 0, 0, 0, 0, 0), rejected=True)
    checked(transport, 1, (0, 0, 0, 0, 0, 0), bytes(PAGE_BYTES), rejected=True)
    checked(transport, 0, (0, 0, 0, 0, 0, 0))
    _, _, after = inspect(transport)
    if after[0] - before[0] != len(cases) or after[1] - before[1] != len(cases):
        raise AssertionError(f"mapping accounting: before={before}, after={after}")
    print(f"[native-kv] PASS controller={controller}, ns={namespace}: "
          f"{len(cases)} physical pages; token/channel addressing, isolation, "
          "partial groups, rejected overwrite/stale/out-of-range/LBA commands")
    return task


def retry_buffer(transport, opcode, fields, payload=None):
    deadline = time.monotonic() + 30
    while True:
        status, core, data = transport(opcode, fields, payload)
        if not status:
            return data
        if core != 5 or time.monotonic() >= deadline:
            raise AssertionError(f"buffer opcode={opcode:#x}: status={status:#x}, core={core}")
        time.sleep(0.001)


def buffer_row(stream, token, value=False):
    return struct.pack("<128H", *[
        ((stream * 79 + token * 131 + channel) & 0xFFFF) ^ (0x1357 if value else 0)
        for channel in range(128)
    ])


def run_buffer_contract(transport):
    inspect(transport)
    task = secrets.randbits(32)
    packet = struct.pack("<8IQI", 2, 1, 1, 128, 600, 8, 2, PAGE_BYTES, 1024 * 1024, 32)
    checked(transport, BUFFER_REGISTER, coordinates(task), packet.ljust(PAGE_BYTES, b"\0"))

    def stats(stream):
        return struct.unpack_from("<11Q", checked(transport, BUFFER_STATUS, coordinates(task, batch=stream)))

    def verify(stream, token):
        for kind in (0, 2):
            data = checked(transport, BUFFER_READ, coordinates(task, batch=stream, token=token, kind=kind))
            if data[:256] != buffer_row(stream, token, kind == 2):
                raise AssertionError(f"buffer row mismatch: batch={stream}, token={token}, kind={kind}")
        expected = buffer_row(stream, token)
        for channel in (0, 7, 8, 127):
            data = checked(transport, BUFFER_READ,
                           coordinates(task, batch=stream, token=token, kind=1, channel=channel))
            if data[:2] != expected[channel * 2:channel * 2 + 2]:
                raise AssertionError(f"channel-K mismatch: batch={stream}, token={token}, channel={channel}")

    accepted = [0, 0]
    memory_bytes = stats(0)[4]
    for boundary in (15, 16, 253, 256, 259, 531):
        for stream in range(2):
            while accepted[stream] < boundary:
                first = accepted[stream]
                count = min(8, boundary - first)
                payload = b"".join(buffer_row(stream, token) for token in range(first, first + count))
                payload += b"".join(buffer_row(stream, token, True) for token in range(first, first + count))
                retry_buffer(transport, BUFFER_APPEND, coordinates(task, batch=stream, token=first, valid=count),
                             payload.ljust(PAGE_BYTES, b"\0"))
                accepted[stream] += count
            verify(stream, 0)
            verify(stream, boundary - 1)
        if boundary == 15:
            snapshot = stats(0)
            if snapshot[5] or snapshot[6] or any(snapshot[1:4]):
                raise AssertionError(f"incomplete groups unexpectedly used NVM: {snapshot}")
    checked(transport, WRITE, coordinates(task, valid=16), bytes(PAGE_BYTES), rejected=True)
    checked(transport, READ, coordinates(task), rejected=True)
    checked(transport, BUFFER_APPEND, coordinates(task, valid=1), bytes(PAGE_BYTES), rejected=True)
    checked(transport, BUFFER_READ, coordinates(task, generation=2), rejected=True)
    checked(transport, BUFFER_READ, coordinates(task, token=531), rejected=True)
    retry_buffer(transport, BUFFER_DRAIN, coordinates(task))
    for stream in range(2):
        snapshot = stats(stream)
        if snapshot[:4] != (531, 531, 531, 531) or snapshot[4] != memory_bytes or \
                snapshot[5] != 950272 or not snapshot[9] or snapshot[10]:
            raise AssertionError(f"buffer drain/accounting mismatch: {snapshot}")
        for token in (0, 14, 15, 16, 252, 255, 256, 258, 512, 530):
            verify(stream, token)
    checked(transport, 0, (0, 0, 0, 0, 0, 0))
    print(f"[native-buffer] PASS: 2 streams x 531 tokens, K dual layout/V, "
          f"15/16 and 255/256 boundaries, ring reuse, partial drain, "
          f"bounded_memory={memory_bytes}, NVM_write_bytes=950272; no OPT/Attention")


def main():
    parser = argparse.ArgumentParser(description="Destructive only to a fresh native KV test namespace; never format it")
    parser.add_argument("--device", required=True)
    parser.add_argument("--buffer-test", action="store_true")
    args = parser.parse_args()
    transport = LinuxTransport(args.device)
    try:
        if args.buffer_test:
            run_buffer_contract(transport)
        else:
            run_contract(transport)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
