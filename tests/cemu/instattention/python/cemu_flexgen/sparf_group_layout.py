from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class GroupPageAddress:
    page_index: int
    byte_offset: int


@dataclass(frozen=True)
class SparfGroupLayout:
    """Host reference for head-local grouped pages, not an FTL or device backend."""

    head_dim: int
    channels_per_page: int
    page_bytes: int = 4096
    dtype: object = np.float16

    def __post_init__(self):
        for name in ("head_dim", "channels_per_page", "page_bytes"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), 1))
        dtype = np.dtype(self.dtype)
        if dtype not in (np.dtype("<f2"), np.dtype("<f4")):
            raise ValueError("dtype must be little-endian float16 or float32")
        object.__setattr__(self, "dtype", dtype)
        if self.page_bytes % 512:
            raise ValueError("page_bytes must be a multiple of 512")
        if self.head_dim % self.channels_per_page:
            raise ValueError("head_dim must be divisible by channels_per_page")
        for width in (self.head_dim, self.channels_per_page):
            if self.page_bytes % (width * self.dtype.itemsize):
                raise ValueError("a page must hold an integral number of token entries")

    @property
    def token_group_tokens(self):
        return self.page_bytes // (self.head_dim * self.dtype.itemsize)

    @property
    def channel_group_tokens(self):
        return self.page_bytes // (self.channels_per_page * self.dtype.itemsize)

    @property
    def channel_group_pages(self):
        return self.head_dim // self.channels_per_page

    def _coordinate(self, token, channel):
        token = _integer("token", token, 0)
        channel = _integer("channel", channel, 0)
        if channel >= self.head_dim:
            raise ValueError("channel exceeds head_dim")
        return token, channel

    def token_address(self, token, channel):
        token, channel = self._coordinate(token, channel)
        group, position = divmod(token, self.token_group_tokens)
        return GroupPageAddress(group, (position * self.head_dim + channel) * self.dtype.itemsize)

    def channel_address(self, token, channel):
        token, channel = self._coordinate(token, channel)
        group, position = divmod(token, self.channel_group_tokens)
        channel_group, column = divmod(channel, self.channels_per_page)
        return GroupPageAddress(
            group * self.channel_group_pages + channel_group,
            (column * self.channel_group_tokens + position) * self.dtype.itemsize,
        )

    def _values(self, values, capacity):
        values = np.asarray(values)
        if values.dtype != self.dtype:
            raise ValueError("values dtype must match the layout; no implicit conversion")
        if values.ndim != 2 or values.shape[1] != self.head_dim:
            raise ValueError("values must have shape [tokens, head_dim]")
        if not 0 < values.shape[0] <= capacity:
            raise ValueError("values must contain between one token and one complete group")
        return values

    def pack_token_page(self, values):
        values = self._values(values, self.token_group_tokens)
        page = np.zeros((self.token_group_tokens, self.head_dim), dtype=self.dtype)
        page[:values.shape[0]] = values
        return page.tobytes()

    def pack_channel_group(self, values):
        values = self._values(values, self.channel_group_tokens)
        pages = np.zeros(
            (self.channel_group_pages, self.channels_per_page, self.channel_group_tokens),
            dtype=self.dtype,
        )
        pages[:, :, :values.shape[0]] = values.T.reshape(
            self.channel_group_pages, self.channels_per_page, values.shape[0],
        )
        return pages.tobytes()

    def _decode(self, data, valid_tokens, capacity, byte_count):
        valid_tokens = _integer("valid_tokens", valid_tokens, 1)
        if valid_tokens > capacity:
            raise ValueError("valid_tokens exceeds group capacity")
        view = memoryview(data)
        if view.nbytes != byte_count:
            raise ValueError("incorrect group byte length")
        return np.frombuffer(view, dtype=self.dtype), valid_tokens

    def unpack_token_page(self, data, valid_tokens):
        entries, valid_tokens = self._decode(
            data, valid_tokens, self.token_group_tokens, self.page_bytes,
        )
        return entries.reshape(self.token_group_tokens, self.head_dim)[:valid_tokens].copy()

    def unpack_channel_group(self, data, valid_tokens):
        entries, valid_tokens = self._decode(
            data, valid_tokens, self.channel_group_tokens,
            self.channel_group_pages * self.page_bytes,
        )
        return entries.reshape(self.head_dim, self.channel_group_tokens)[:, :valid_tokens].T.copy()
