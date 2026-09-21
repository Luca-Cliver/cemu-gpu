import fcntl
import os
from pathlib import Path
import secrets
import stat
import struct
from contextlib import contextmanager
from threading import RLock

import numpy as np

from .sparf_device_writeback import initialized_extents, initialize_unwritten_extents
from .sparf_group_capacity import GROUP_IO_BYTES, GROUP_STATUS_BYTES


COMMAND = 0x43454D5553475250
HEADER = struct.Struct("<8I4Q")
EXTENT = struct.Struct("<QQ")
REGISTER, PUT, READ_TOKENS, READ_CHANNELS, CLOSE = range(1, 6)
REGISTER_BUFFER, APPEND, DRAIN, STATUS = range(6, 10)
APPEND_BATCH = 10
STATS = 11
TOKEN_K, CHANNEL_K, TOKEN_V = range(3)


def registration(layout, extents, cookie, buffer_streams=0, write_batch_pages=0):
    return HEADER.pack(1, REGISTER_BUFFER if buffer_streams else REGISTER, 0, layout.head_dim, layout.channels_per_page,
                       layout.dtype.itemsize, layout.page_bytes, len(extents), buffer_streams, 0, write_batch_pages, cookie) + b"".join(
                           EXTENT.pack(*extent) for extent in extents)


def command(operation, cookie, kind=0, stream=0, position=0, count=0, argument=0, indices=()):
    return HEADER.pack(1, operation, kind, 0, 0, 0, 0, count, stream, position, argument, cookie) + b"".join(
        struct.pack("<Q", index) for index in indices)


class SparfGroupedDevice:
    """Device-owned grouped IO and bounded KV tails.

    The CSD session has one shared MR0.  Every write/read transaction therefore
    holds the lock from filling MR0 through execute and status/output transfer.
    Attention workers remain independently schedulable; only this narrow
    transport is serialized.
    """

    def __init__(self, layout, pool_path, pool_bytes, fdm_dir, program_path,
                 control_path="/dev/nvme0c3", namespace_path="/dev/ng0n3", nvm_path="/dev/nvme0n1",
                 buffer_streams=0, write_batch_pages=0):
        if pool_bytes <= 0 or pool_bytes % layout.page_bytes:
            raise ValueError("pool_bytes must contain whole pages")
        self.layout = layout
        if not isinstance(buffer_streams, int) or not 0 <= buffer_streams <= (1 << 20):
            raise ValueError("buffer_streams must be between 0 and 1048576")
        self.buffer_streams = buffer_streams
        if (type(write_batch_pages) is not int or write_batch_pages < 0 or
                (write_batch_pages and (not buffer_streams or write_batch_pages < layout.channel_group_pages)) or
                write_batch_pages * layout.page_bytes > 64 * 1024 * 1024):
            raise ValueError("invalid grouped write batch page capacity")
        self.write_batch_pages = write_batch_pages
        self.tail_bytes = buffer_streams * 2 * (layout.channel_group_tokens + layout.token_group_tokens) * (
            layout.head_dim * layout.dtype.itemsize) + (write_batch_pages or layout.channel_group_pages) * layout.page_bytes
        self.pool_path = Path(pool_path)
        self.pool_bytes = pool_bytes
        self.fdm_dir = Path(fdm_dir)
        self.program_path = program_path
        self.control_path = control_path
        self.namespace_path = namespace_path
        self.nvm_path = nvm_path
        self.cookie = secrets.randbits(64) or 1
        self.buffer_bytes = max(
            4096,
            GROUP_IO_BYTES if buffer_streams else layout.channel_group_pages * layout.page_bytes,
        )
        self.device = None
        self.descriptor = -1
        self.registered = False
        self.last_result = None
        self._io_lock = RLock()
        self._failure = None

    def open(self):
        from .cemu_device import CemuDevice, RangeSpec

        if self.device is not None:
            return self
        self._failure = None
        namespace = os.stat(self.nvm_path)
        if not stat.S_ISBLK(namespace.st_mode) or self.pool_path.parent.stat().st_dev != namespace.st_rdev:
            raise ValueError("pool must be a private file on the selected CEMU NVM namespace")
        try:
            self.descriptor = os.open(self.pool_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.posix_fallocate(self.descriptor, 0, self.pool_bytes)
            initialize_unwritten_extents(self.descriptor, self.pool_bytes)
            os.fsync(self.descriptor)
            extents = initialized_extents(self.descriptor, self.pool_bytes)
            os.posix_fadvise(self.descriptor, 0, self.pool_bytes, os.POSIX_FADV_DONTNEED)
            ranges = [RangeSpec(str(self.fdm_dir / "group_io"), self.buffer_bytes),
                      RangeSpec(str(self.fdm_dir / "group_status"), GROUP_STATUS_BYTES)]
            if self.buffer_streams:
                ranges.append(RangeSpec(str(self.fdm_dir / "group_tail"), self.tail_bytes))
            self.device = CemuDevice(
                f"sparf_group_{self.cookie:x}", self.program_path, "sparf_attention",
                tuple(ranges),
                control_path=self.control_path, namespace_path=self.namespace_path,
            )
            self.device.open()
            self._execute(registration(self.layout, extents, self.cookie, self.buffer_streams, self.write_batch_pages))
            self.registered = True
        except Exception:
            self.close()
            raise
        return self

    def _execute(self, metadata, allow_failed=False):
        if self.device is None:
            raise RuntimeError("group device is not open")
        if self._failure is not None and not allow_failed:
            raise RuntimeError("grouped device is poisoned after a prior command failure") from self._failure
        try:
            if self.device.execute(cparam1=COMMAND, metadata=metadata):
                raise RuntimeError("group command failed")
            self.last_result = self.device.read_tensor(1, (20 if self.buffer_streams else 12,), np.uint64)
            if int(self.last_result[8]) != 1:
                raise RuntimeError("group result version mismatch; rebuild and restart QEMU")
            return self.last_result
        except BaseException as error:
            if not allow_failed:
                self._failure = error
            raise

    def put(self, kind, stream, start_token, values):
        if self.buffer_streams:
            raise ValueError("buffered sessions require APPEND, not PUT")
        if kind not in (TOKEN_K, CHANNEL_K, TOKEN_V):
            raise ValueError("unknown grouped layout")
        values = np.asarray(values)
        capacity = self.layout.channel_group_tokens if kind == CHANNEL_K else self.layout.token_group_tokens
        if values.dtype != self.layout.dtype or values.shape != (capacity, self.layout.head_dim):
            raise ValueError("PUT requires one full group with the registered dtype")
        if start_token < 0 or start_token % capacity:
            raise ValueError("PUT start_token must be group aligned")
        with self._mutation():
            self.device.write_tensor(0, np.ascontiguousarray(values))
            return self._execute(command(PUT, self.cookie, kind, stream, start_token, capacity))

    def append(self, stream, start_token, key, value):
        self._validate_stream(stream)
        key, value = np.asarray(key), np.asarray(value)
        if (key.ndim != 2 or key.shape[1] != self.layout.head_dim or value.shape != key.shape or
                key.dtype != self.layout.dtype or value.dtype != self.layout.dtype or not len(key)):
            raise ValueError("APPEND requires matching nonempty [tokens, head_dim] K/V arrays")
        if type(start_token) is not int or start_token < 0 or start_token + len(key) > 2**64 - 1:
            raise ValueError("APPEND token range must fit uint64")
        capacity = self.buffer_bytes // (2 * self.layout.head_dim * self.layout.dtype.itemsize)
        if not capacity:
            raise ValueError("IO buffer cannot hold one K/V entry")
        with self._mutation():
            for offset in range(0, len(key), capacity):
                count = min(capacity, len(key) - offset)
                packet = np.stack((key[offset:offset + count], value[offset:offset + count]))
                self.device.write_tensor(0, packet)
                self._execute(command(APPEND, self.cookie, stream=stream,
                                      position=start_token + offset, count=count))
            return self.last_result

    def progress(self, stream):
        self._validate_stream(stream)
        with self._io_lock:
            self._require_healthy()
            result = self._execute(command(STATUS, self.cookie, stream=stream))
        return dict(zip(("accepted", "token_k", "channel_k", "token_v", "sealed", "failed", "write_bytes", "buffer_bytes"),
                        map(int, result[12:])))

    def append_batch(self, stream, start_token, keys, values):
        self._validate_stream(stream)
        keys, values = np.asarray(keys), np.asarray(values)
        if (keys.ndim != 3 or values.shape != keys.shape or keys.dtype != self.layout.dtype or
                values.dtype != self.layout.dtype or keys.shape[2] != self.layout.head_dim or
                not keys.shape[0] or not keys.shape[1] or stream + keys.shape[1] > self.buffer_streams or
                type(start_token) is not int or start_token < 0 or start_token + keys.shape[0] > 2**64 - 1):
            raise ValueError("APPEND_BATCH requires valid [tokens, streams, head_dim] K/V arrays")
        entry_bytes = 2 * self.layout.head_dim * self.layout.dtype.itemsize
        if self.buffer_bytes < entry_bytes:
            raise ValueError("IO buffer cannot hold one K/V entry")
        token_tile = min(self.layout.token_group_tokens, self.buffer_bytes // entry_bytes)
        with self._mutation():
            for position in range(0, len(keys), token_tile):
                count = min(token_tile, len(keys) - position)
                stream_tile = self.buffer_bytes // (count * entry_bytes)
                for first in range(0, keys.shape[1], stream_tile):
                    stop = min(first + stream_tile, keys.shape[1])
                    packet = np.stack((keys[position:position + count, first:stop],
                                       values[position:position + count, first:stop]))
                    self.device.write_tensor(0, packet)
                    self._execute(command(APPEND_BATCH, self.cookie, stream=stream + first,
                                          position=start_token + position, count=count, argument=stop - first))
            return self.last_result

    def drain(self):
        self._validate_stream(0)
        with self._io_lock:
            self._require_healthy()
            return self._execute(command(DRAIN, self.cookie))

    def statistics(self):
        self._validate_stream(0)
        with self._io_lock:
            self._require_healthy()
            self._execute(command(STATS, self.cookie))
            values = self.device.read_tensor(0, (5,), np.uint64)
        return dict(zip(("write_batches", "max_write_pages", "append_commands", "appended_streams", "appended_tokens"),
                        map(int, values)))

    def _validate_stream(self, stream):
        if type(stream) is not int or not self.buffer_streams or not 0 <= stream < self.buffer_streams:
            raise ValueError("stream is outside the buffered session")

    def read_tokens(self, kind, stream, tokens):
        if kind not in (TOKEN_K, TOKEN_V):
            raise ValueError("token reads require token-K or token-V")
        tokens = tuple(tokens)
        self._validate_read(len(tokens), len(tokens) * self.layout.head_dim)
        with self._io_lock:
            self._require_healthy()
            self._execute(command(READ_TOKENS, self.cookie, kind, stream,
                                  count=len(tokens), indices=tokens))
            return self.device.read_tensor(0, (len(tokens), self.layout.head_dim), self.layout.dtype)

    def read_channels(self, stream, start_token, token_count, channels):
        channels = tuple(channels)
        self._validate_read(len(channels), len(channels) * token_count)
        with self._io_lock:
            self._require_healthy()
            self._execute(command(READ_CHANNELS, self.cookie, CHANNEL_K, stream, start_token,
                                  len(channels), token_count, channels))
            return self.device.read_tensor(0, (len(channels), token_count), self.layout.dtype)

    def flush(self):
        """Non-terminal barrier.

        APPEND completion means the device owns a stable tail copy.  Unfilled
        groups intentionally stay in FDM so Decode can continue appending.
        """
        with self._io_lock:
            self._require_healthy()

    @contextmanager
    def _mutation(self):
        # One append may span many commands. Even an MR0 write failure after
        # a successful tile leaves the caller's accepted prefix uncertain.
        with self._io_lock:
            self._require_healthy()
            try:
                yield
            except BaseException as error:
                self._failure = error
                raise

    def _validate_read(self, count, elements):
        if not 0 < count <= 4096 or elements <= 0 or elements * self.layout.dtype.itemsize > self.buffer_bytes:
            raise ValueError("read exceeds the registered IO buffer or index count limit")

    def _require_healthy(self):
        if self.device is None:
            raise RuntimeError("group device is not open")
        if self._failure is not None:
            raise RuntimeError("grouped device is poisoned after a prior command failure") from self._failure

    def check_ready(self):
        """Reject use of a shared session whose accepted prefix is uncertain."""
        with self._io_lock:
            self._require_healthy()

    def close(self):
        with self._io_lock:
            try:
                if self.device is not None:
                    try:
                        if self.registered:
                            self._execute(command(CLOSE, self.cookie), allow_failed=True)
                    finally:
                        self.registered = False
                        try:
                            self.device.close()
                        finally:
                            self.device = None
                            self._failure = None
            finally:
                if self.descriptor >= 0:
                    os.close(self.descriptor)
                    self.descriptor = -1

    def __enter__(self):
        return self.open()

    def __exit__(self, *arguments):
        self.close()
