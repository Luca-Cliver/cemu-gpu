"""Select the Guest transport without changing device commands or dependencies."""

import importlib
import os


def io_backend():
    backend = os.environ.get("CEMU_IO_BACKEND", "ioctl")
    if backend not in ("ioctl", "uring"):
        raise ValueError("CEMU_IO_BACKEND must be ioctl or uring")
    if backend == "uring":
        uring_bindings()
    return backend


def uring_bindings():
    try:
        module = importlib.import_module("_cemu_client")
    except ImportError as error:
        raise RuntimeError("build _cemu_client with io_uring support and add it to PYTHONPATH") from error
    if not getattr(module, "io_uring_available", lambda: False)():
        raise RuntimeError("rebuild _cemu_client with liburing >= 2.2 and NVMe uring headers; no ioctl fallback")
    return module


def copy_file_range(source_fd, destination_fd, size, *, offset_src, offset_dst):
    if io_backend() == "uring":
        return uring_bindings().uring_copy_file_range(
            source_fd, destination_fd, min(size, 0x7fffffff), offset_src, offset_dst
        )
    return os.copy_file_range(source_fd, destination_fd, size,
                              offset_src=offset_src, offset_dst=offset_dst)
