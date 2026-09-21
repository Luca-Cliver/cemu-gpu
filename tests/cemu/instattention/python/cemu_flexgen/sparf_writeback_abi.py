import bisect
import struct

COMMAND = 0x43454D555357424B
HEADER = struct.Struct("<IIQ")
SPAN = struct.Struct("<QQIIII")
MATRIX = struct.Struct("<QQIIIIIIII")
RESULT = struct.Struct("<8Q")
DIAGNOSTIC_FIELDS = (
    "version", "plan_host_ns", "queue_host_ns", "prepare_host_ns",
    "read_service_host_ns", "read_target_wait_host_ns", "patch_host_ns",
    "write_service_host_ns", "write_target_wait_host_ns", "metadata_bytes", "expanded_spans",
)
MAX_SPANS = 65536


def pack_writeback(buffer, layer, start, count, maps, compact=False, profile=False):
    layout = buffer.layout
    layout.token_head_offset(layer, 0, 0, start)
    if count <= 0 or count > buffer.capacity or start + count > layout.config.max_seq_len:
        raise ValueError("invalid writeback group")
    payload = bytearray(HEADER.size)
    descriptor = MATRIX if compact else SPAN
    mapped = []
    for extents, size in zip(maps, (layout.token_file_size, layout.channel_file_size, layout.token_file_size)):
        offsets, physical, total = [], [], 0
        for slba, nlb in extents:
            if slba < 0 or nlb <= 0:
                raise ValueError("invalid extent")
            offsets.append(total)
            physical.append(slba * 512)
            total += nlb * 512
        if total != size:
            raise ValueError("incomplete file mapping")
        mapped.append((offsets + [total], physical))
    if len(mapped) != 3:
        raise ValueError("three file maps required")

    def append(destination, source, size, records, destination_stride,
               rows=1, source_row_stride=0, destination_row_stride=0):
        if (len(payload) - HEADER.size) // descriptor.size >= MAX_SPANS:
            raise ValueError("writeback metadata exceeds span limit; use a smaller group")
        fields = (destination, source, size, records, buffer.record_bytes, destination_stride)
        if compact:
            fields += (rows, source_row_stride, destination_row_stride, 0)
        payload.extend(descriptor.pack(*fields))

    def emit(file, destination, source, size, records, destination_stride):
        offsets, physical = mapped[file]
        while records:
            extent = bisect.bisect_right(offsets, destination) - 1
            if extent < 0 or extent >= len(physical):
                raise ValueError("destination outside mapping")
            available = offsets[extent + 1] - destination
            if available < size:
                emit(file, destination, source, available, 1, available)
                emit(file, destination + available, source + available, size - available, 1, size - available)
                run = 1
            else:
                run = min(records, 1 + (available - size) // destination_stride)
                append(physical[extent] + destination - offsets[extent], source,
                       size, run, destination_stride)
            destination += run * destination_stride
            source += run * buffer.record_bytes
            records -= run

    def emit_channels(destination, source, records):
        rows = layout.config.head_dim
        offsets, physical = mapped[1]
        size = layout.element_size
        while rows:
            extent = bisect.bisect_right(offsets, destination) - 1
            if extent < 0 or extent >= len(physical):
                raise ValueError("channel destination outside mapping")
            available = offsets[extent + 1] - destination
            footprint = records * size
            if available < footprint:
                emit(1, destination, source, size, records, size)
                run = 1
            else:
                run = min(rows, 1 + (available - footprint) // layout.channel_stride)
                append(physical[extent] + destination - offsets[extent], source,
                       size, records, size, run, size, layout.channel_stride)
            destination += run * layout.channel_stride
            source += run * size
            rows -= run

    remaining, token = count, start
    while remaining:
        run = min(remaining, buffer.capacity - token % buffer.capacity)
        base = buffer._offset(layer, token)
        for batch in range(layout.config.batch_size):
            for head in range(layout.config.num_kv_heads):
                source = base + (batch * layout.config.num_kv_heads + head) * layout.head_bytes
                destination = layout.token_head_offset(layer, batch, head, token)
                emit(0, destination, source, layout.head_bytes, run, layout.head_bytes)
                emit(2, destination, source + buffer.token_bytes, layout.head_bytes, run, layout.head_bytes)
                if compact:
                    emit_channels(layout.channel_offset(layer, batch, head, 0, token), source, run)
                else:
                    for dimension in range(layout.config.head_dim):
                        destination = layout.channel_offset(layer, batch, head, dimension, token)
                        emit(1, destination, source + dimension * layout.element_size,
                             layout.element_size, run, layout.element_size)
        token += run
        remaining -= run
    HEADER.pack_into(payload, 0, 2 if compact else 1,
                     (len(payload) - HEADER.size) // descriptor.size, int(profile))
    return bytes(payload)
