#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "fdm_kv_receive.h"

static bool bitmap_matches(const FDMKV_ReceiveState_t *receive,
                           uint64_t offset, uint64_t bytes, bool received)
{
    uint64_t first = offset / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    uint64_t count = bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    for (uint64_t unit = first; unit < first + count; unit++) {
        bool present = (receive->received_bitmap[unit / 8] >> (unit % 8)) & 1;
        if (present != received) {
            return false;
        }
    }
    return true;
}

FDMKV_ReceiveResult fdm_kv_receive_check(const FDMKV_Buffer_t *buffer,
                                       uint64_t offset, uint64_t bytes)
{
    if (!buffer || !buffer->initialized || !bytes ||
        offset % CEMU_FDM_KV_RECEIVE_UNIT_BYTES ||
        bytes % CEMU_FDM_KV_RECEIVE_UNIT_BYTES ||
        offset > buffer->input_bytes || bytes > buffer->input_bytes - offset) {
        return FDM_KV_RECEIVE_INVALID;
    }
    const FDMKV_ReceiveState_t *receive = buffer->receive;
    if (buffer->failed || buffer->draining || buffer->sealed ||
        (receive->state != FDM_KV_RX_EMPTY && receive->state != FDM_KV_RX_RECEIVING)) {
        return FDM_KV_RECEIVE_BUSY;
    }
    if (receive->header_valid &&
        (offset > receive->header.wire_bytes || bytes > receive->header.wire_bytes - offset)) {
        return FDM_KV_RECEIVE_INVALID;
    }
    if (!bitmap_matches(receive, offset, bytes, false)) {
        return FDM_KV_RECEIVE_BUSY;
    }
    return FDM_KV_RECEIVE_ACCEPTED;
}

static bool bytes_zero(const uint8_t *data, uint64_t bytes)
{
    for (uint64_t index = 0; index < bytes; index++) {
        if (data[index]) {
            return false;
        }
    }
    return true;
}

static bool receive_header(FDMKV_Buffer_t *buffer)
{
    FDMKV_AppendHeader_t wire;
    memcpy(&wire, buffer->input, sizeof(wire));
    FDMKV_AppendHeader_t header = {
        .magic = le32_to_cpu(wire.magic),
        .version = le32_to_cpu(wire.version),
        .header_bytes = le32_to_cpu(wire.header_bytes),
        .operation = le32_to_cpu(wire.operation),
        .session_id = le64_to_cpu(wire.session_id),
        .generation = le64_to_cpu(wire.generation),
        .request_id = le64_to_cpu(wire.request_id),
        .wire_bytes = le64_to_cpu(wire.wire_bytes),
        .payload_bytes = le64_to_cpu(wire.payload_bytes),
        .batch = le32_to_cpu(wire.batch),
        .layer = le32_to_cpu(wire.layer),
        .head_first = le32_to_cpu(wire.head_first),
        .head_count = le32_to_cpu(wire.head_count),
        .token_first = le64_to_cpu(wire.token_first),
        .token_count = le32_to_cpu(wire.token_count),
        .reserved = le32_to_cpu(wire.reserved),
    };
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    if (header.magic != CEMU_FDM_KV_PACKET_MAGIC ||
        header.version != CEMU_FDM_KV_PACKET_VERSION ||
        header.header_bytes != CEMU_FDM_KV_PACKET_HEADER_BYTES ||
        (header.operation != FDM_KV_OP_APPEND && header.operation != FDM_KV_OP_DRAIN) ||
        header.reserved ||
        header.session_id != buffer->identity.session_id ||
        header.generation != buffer->identity.generation ||
        header.request_id != buffer->receive->next_request_id ||
        header.request_id == UINT64_MAX ||
        !bytes_zero(buffer->input + sizeof(wire),
                    CEMU_FDM_KV_PACKET_HEADER_BYTES - sizeof(wire))) {
        return false;
    }
    if (header.operation == FDM_KV_OP_DRAIN) {
        if (header.wire_bytes != CEMU_FDM_KV_PACKET_HEADER_BYTES || header.payload_bytes ||
            header.batch || header.layer || header.head_first || header.head_count ||
            header.token_first || header.token_count ||
            !bitmap_matches(buffer->receive, header.wire_bytes,
                            buffer->input_bytes - header.wire_bytes, false)) {
            return false;
        }
        buffer->receive->header = header;
        buffer->receive->header_valid = 1;
        return true;
    }
    if (header.batch >= geometry->batches || header.layer >= geometry->layers ||
        header.head_first >= geometry->heads || !header.head_count ||
        header.head_count > geometry->heads - header.head_first ||
        !header.token_count || header.token_count > buffer->append_max_tokens ||
        header.token_first > geometry->max_tokens ||
        header.token_count > geometry->max_tokens - header.token_first) {
        return false;
    }
    uint64_t rows = (uint64_t)header.head_count * header.token_count;
    uint64_t row_pair_bytes = 2ULL * buffer->row_bytes;
    if (!row_pair_bytes || rows > UINT64_MAX / row_pair_bytes ||
        header.payload_bytes != rows * row_pair_bytes ||
        header.payload_bytes > buffer->input_bytes - CEMU_FDM_KV_PACKET_HEADER_BYTES) {
        return false;
    }
    uint64_t packet_bytes = CEMU_FDM_KV_PACKET_HEADER_BYTES + header.payload_bytes;
    uint64_t padding = (512 - packet_bytes % 512) % 512;
    if (padding > buffer->input_bytes - packet_bytes ||
        header.wire_bytes != packet_bytes + padding ||
        !bitmap_matches(buffer->receive, header.wire_bytes,
                        buffer->input_bytes - header.wire_bytes, false)) {
        return false;
    }
    buffer->receive->header = header;
    buffer->receive->header_valid = 1;
    return true;
}

void fdm_kv_receive_fail(FDMKV_Buffer_t *buffer)
{
    buffer->receive->state = FDM_KV_RX_FAILED;
    buffer->failed = 1;
}

bool fdm_kv_receive_release(FDMKV_Buffer_t *buffer)
{
    if (!buffer || !buffer->initialized || buffer->failed ||
        buffer->receive->state != FDM_KV_RX_AGGREGATED ||
        buffer->receive->next_request_id == UINT64_MAX) {
        return false;
    }
    FDMKV_ReceiveState_t *receive = buffer->receive;
    memset(receive->received_bitmap, 0, receive->bitmap_bytes);
    memset(&receive->header, 0, sizeof(receive->header));
    receive->header_valid = 0;
    receive->max_write_ready_ns = 0;
    receive->next_request_id++;
    receive->state = FDM_KV_RX_EMPTY;
    return true;
}

FDMKV_ReceiveResult fdm_kv_receive_record(FDMKV_Buffer_t *buffer,
                                        uint64_t offset, uint64_t bytes,
                                        uint64_t ready_ns)
{
    FDMKV_ReceiveResult result = fdm_kv_receive_check(buffer, offset, bytes);
    if (result != FDM_KV_RECEIVE_ACCEPTED) {
        return result;
    }
    FDMKV_ReceiveState_t *receive = buffer->receive;
    uint64_t first = offset / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    uint64_t count = bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    for (uint64_t unit = first; unit < first + count; unit++) {
        receive->received_bitmap[unit / 8] |= 1U << (unit % 8);
    }
    receive->state = FDM_KV_RX_RECEIVING;
    receive->max_write_ready_ns = MAX(receive->max_write_ready_ns, ready_ns);
    if (!receive->header_valid &&
        bitmap_matches(receive, 0, CEMU_FDM_KV_PACKET_HEADER_BYTES, true) &&
        !receive_header(buffer)) {
        fdm_kv_receive_fail(buffer);
        return FDM_KV_RECEIVE_INVALID;
    }
    if (receive->header_valid && bitmap_matches(receive, 0, receive->header.wire_bytes, true)) {
        uint64_t payload_end = CEMU_FDM_KV_PACKET_HEADER_BYTES + receive->header.payload_bytes;
        if (!bytes_zero(buffer->input + payload_end, receive->header.wire_bytes - payload_end)) {
            fdm_kv_receive_fail(buffer);
            return FDM_KV_RECEIVE_INVALID;
        }
        receive->state = FDM_KV_RX_READY;
        return FDM_KV_RECEIVE_READY;
    }
    return FDM_KV_RECEIVE_ACCEPTED;
}
