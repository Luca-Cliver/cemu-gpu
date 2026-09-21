import fcntl
import os
import struct
import time
from threading import Lock

import numpy as np

from .sparf_write_buffer import _aligned_buffer
from .sparf_writeback_abi import COMMAND, DIAGNOSTIC_FIELDS, pack_writeback
from phase_profiler import profile_scope


def file_extents(descriptor, size):
    header = struct.Struct("<QQIIII")
    extent = struct.Struct("<QQQQQIIII")
    covered = 0
    while covered < size:
        packet = bytearray(header.size + 256 * extent.size)
        header.pack_into(packet, 0, covered, size - covered, 1, 0, 256, 0)
        fcntl.ioctl(descriptor, 0xC020660B, packet, True)
        count = header.unpack_from(packet)[3]
        if not 0 < count <= 256:
            raise ValueError("missing initialized file extents")
        for index in range(count):
            logical, physical, length, _, _, flags, _, _, _ = extent.unpack_from(packet, header.size + index * extent.size)
            if logical != covered or flags & ~(1 | 0x800) or physical % 512 or length % 512 or not length:
                raise ValueError("writeback needs private, contiguous logical extents; holes/shared extents rejected")
            length = min(length, size - covered)
            if not length or length % 512:
                raise ValueError("invalid file size or extra extent")
            yield logical, physical, length, flags
            covered += length


def initialized_extents(descriptor, size):
    result = []
    for logical, physical, length, flags in file_extents(descriptor, size):
        if flags & 0x800:
            raise ValueError("writeback needs initialized extents; unwritten extents rejected")
        result.append((physical // 512, length // 512))
    return tuple(result)


def initialize_unwritten_extents(descriptor, size):
    """Materialize only allocated unwritten ranges before device ownership."""
    pending = [(logical, length) for logical, physical, length, flags
               in file_extents(descriptor, size) if flags & 0x800]
    if not pending:
        return 0
    zeros = bytes(1024 * 1024)
    written = 0
    for offset, remaining in pending:
        while remaining:
            count = os.pwrite(descriptor, memoryview(zeros)[:min(remaining, len(zeros))], offset)
            if count <= 0:
                raise OSError("failed to initialize unwritten KV extent")
            offset += count
            remaining -= count
            written += count
    os.fsync(descriptor)
    return written


def direct_read(descriptor, size, offset):
    start = offset // 4096 * 4096
    length = (offset + size - start + 4095) // 4096 * 4096
    buffer = _aligned_buffer(length)
    received = os.preadv(descriptor, [buffer], start)
    if received < offset - start + size:
        raise EOFError("short direct cache read")
    return buffer[offset - start:offset - start + size].tobytes()


class SparfDeviceWriteback:
    _locks = {}
    _locks_guard = Lock()

    def __init__(self, program_path, control_path="/dev/nvme0c3", namespace_path="/dev/ng0n3",
                 nvm_path="/dev/nvme0n1", metadata_format="compact"):
        if metadata_format not in ("spans", "compact"):
            raise ValueError("writeback metadata format must be spans or compact")
        self.metadata_format = metadata_format
        self.program_path = program_path
        self.control_path = control_path
        self.namespace_path = namespace_path
        self.nvm_path = nvm_path
        self.device = None
        self.maps = None
        self.failed = False
        self.initialized_bytes = 0
        with self._locks_guard:
            self.lock = self._locks.setdefault(os.path.realpath(namespace_path), Lock())

    def acquire(self, buffer):
        from .cemu_device import CemuDevice, RangeSpec

        store = buffer.store
        paths = (store.token_k_path, store.channel_k_path, store.token_v_path)
        names = ("_token_k_fd", "_channel_k_fd", "_token_v_fd")
        sizes = (store.layout.token_file_size, store.layout.channel_file_size, store.layout.token_file_size)
        if any(os.stat(path).st_dev != os.stat(self.nvm_path).st_rdev for path in paths):
            raise ValueError("KV files must belong to the selected CEMU NVM namespace")
        store.publish()
        store._channel_map.close()
        store._channel_map = None
        descriptors = []
        try:
            for path, name, size in zip(paths, names, sizes):
                descriptor = os.open(path, os.O_RDONLY | os.O_DIRECT)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                original = getattr(store, name)
                self.initialized_bytes += initialize_unwritten_extents(original, size)
                os.posix_fadvise(original, 0, size, os.POSIX_FADV_DONTNEED)
            self.maps = tuple(initialized_extents(fd, size) for fd, size in zip(descriptors, sizes))
            physical = sorted((slba, slba + nlb) for mapping in self.maps for slba, nlb in mapping)
            if any(left[1] > right[0] for left, right in zip(physical, physical[1:])):
                raise ValueError("KV destination extents overlap")
            for name, descriptor in zip(names, descriptors):
                os.close(getattr(store, name))
                setattr(store, name, descriptor)
            descriptors.clear()
            store._device_owned = True
            self.device = CemuDevice(
                "sparf_wb_" + buffer.path.name, self.program_path, "sparf_attention",
                (RangeSpec(str(buffer.path), buffer.file_bytes),
                 RangeSpec(str(buffer.path) + "_writeback_status", 4096)),
                control_path=self.control_path, namespace_path=self.namespace_path,
            )
            self.device.open()
        except Exception:
            for descriptor in descriptors:
                os.close(descriptor)
            self.failed = True
            raise

    def writeback(self, buffer, layer, start, count):
        if self.failed or self.device is None:
            raise RuntimeError("device writeback is not available")
        profiler = buffer.profiler
        with profile_scope(profiler, "kv_writeback.metadata_wall"):
            metadata = pack_writeback(buffer, layer, start, count, self.maps,
                                      compact=self.metadata_format == "compact", profile=profiler is not None)
        if profiler is not None:
            profiler.record("kv_writeback.command_metadata", 0, len(metadata))
        waiting = time.perf_counter_ns() if profiler is not None else 0
        try:
            with self.lock:
                if profiler is not None:
                    profiler.record("kv_writeback.lock_wait_wall", time.perf_counter_ns() - waiting)
                with profile_scope(profiler, "kv_writeback.execute_wait_wall"):
                    result = self.device.execute(cparam1=COMMAND, metadata=metadata)
                if result:
                    raise RuntimeError("CEMU device writeback failed")
                if profiler is not None:
                    with profile_scope(profiler, "kv_writeback.status_read_wall"):
                        status = self.device.read_tensor(1, (8 + len(DIAGNOSTIC_FIELDS),), np.uint64)
                    self._record_status(profiler, status)
        except Exception:
            self.failed = True
            raise

    @staticmethod
    def _record_status(profiler, status):
        values = [int(value) for value in status]
        if len(values) != 8 + len(DIAGNOSTIC_FIELDS):
            raise RuntimeError("truncated writeback diagnostics")
        diagnostics = dict(zip(DIAGNOSTIC_FIELDS, values[8:]))
        if diagnostics.get("version") != 1:
            raise RuntimeError("writeback diagnostics unavailable; rebuild and restart QEMU")
        for field, value in diagnostics.items():
            if field.endswith("_host_ns"):
                profiler.record("kv_writeback.device." + field, value)
        profiler.record("kv_writeback.ftl_read_target_ns", values[3] - values[2], values[0])
        profiler.record("kv_writeback.ftl_write_target_ns", values[5] - values[4], values[1])
        profiler.record("kv_writeback.device_payload", 0, values[7])

    def status(self):
        return self.device.read_tensor(1, (8,), np.uint64)

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None
