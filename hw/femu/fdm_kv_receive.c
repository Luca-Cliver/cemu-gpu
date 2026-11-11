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
    bool layer_packet = header.version == CEMU_FDM_KV_LAYER_PACKET_VERSION;
    uint32_t batches = layer_packet ? header.batch_count : 1;
    if (header.magic != CEMU_FDM_KV_PACKET_MAGIC ||
        (!layer_packet && header.version != CEMU_FDM_KV_PACKET_VERSION) ||
        header.header_bytes != CEMU_FDM_KV_PACKET_HEADER_BYTES ||
        (header.operation != FDM_KV_OP_APPEND && header.operation != FDM_KV_OP_DRAIN) ||
        (!layer_packet && header.reserved) ||
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
            header.token_first || header.token_count || header.batch_count ||
            buffer->receive->received_end > header.wire_bytes) {
            return false;
        }
        buffer->receive->header = header;
        buffer->receive->header_valid = 1;
        return true;
    }
    if (header.batch >= geometry->batches || !batches ||
        batches > geometry->batches - header.batch || header.layer >= geometry->layers ||
        header.head_first >= geometry->heads || !header.head_count ||
        header.head_count > geometry->heads - header.head_first ||
        !header.token_count || (!layer_packet && header.token_count > buffer->append_max_tokens) ||
        header.token_first > geometry->max_tokens ||
        header.token_count > geometry->max_tokens - header.token_first) {
        return false;
    }
    uint64_t rows = (uint64_t)header.head_count * header.token_count;
    if (rows > UINT64_MAX / batches) {
        return false;
    }
    rows *= batches;
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
        buffer->receive->received_end > header.wire_bytes) {
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
    receive->received_units = 0;
    receive->received_end = 0;
    receive->aggregate_batch = 0;
    receive->aggregate_tokens = 0;
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
    receive->received_units += count;
    receive->received_end = MAX(receive->received_end, offset + bytes);
    receive->max_write_ready_ns = MAX(receive->max_write_ready_ns, ready_ns);
    if (!receive->header_valid &&
        bitmap_matches(receive, 0, CEMU_FDM_KV_PACKET_HEADER_BYTES, true) &&
        !receive_header(buffer)) {
        fdm_kv_receive_fail(buffer);
        return FDM_KV_RECEIVE_INVALID;
    }
    if (receive->header_valid &&
        receive->received_units == receive->header.wire_bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES) {
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

static bool slot_buffer_valid(const FDMKV_Buffer_t *buffer)
{
    return buffer && buffer->initialized &&
        buffer->version == CEMU_FDM_KV_BUFFER_VERSION && buffer->input_slots &&
        buffer->input_slots <= CEMU_FDM_KV_MAX_INPUT_SLOTS && buffer->slots &&
        buffer->received_heads && !buffer->failed;
}

static FDMKV_Buffer_t slot_view(const FDMKV_Buffer_t *buffer, uint32_t index)
{
    FDMKV_Buffer_t view = *buffer;
    view.input = buffer->slots[index].input;
    view.receive = buffer->slots[index].receive;
    return view;
}

void fdm_kv_slot_fail(FDMKV_Buffer_t *buffer, uint32_t slot_index)
{
    if (buffer && buffer->slots && slot_index < buffer->input_slots) {
        FDMKV_InputSlot_t *slot = &buffer->slots[slot_index];
        slot->state = FDM_KV_SLOT_FAILED;
        slot->dma_active = 0;
        slot->receive->state = FDM_KV_RX_FAILED;
        buffer->failed = 1;
    }
}

FDMKV_ReceiveResult fdm_kv_slot_acquire(FDMKV_Buffer_t *buffer,
                                     uint64_t transfer_id, uint64_t wire_bytes,
                                     uint64_t offset, uint64_t bytes,
                                     uint32_t *slot_index, uint8_t **destination)
{
    uint32_t chosen = UINT32_MAX;
    uint32_t empty = UINT32_MAX;
    bool admission_busy = false;

    if (!slot_buffer_valid(buffer) || !slot_index || !destination || !transfer_id ||
        !wire_bytes || wire_bytes % 512 || wire_bytes > UINT32_MAX ||
        wire_bytes > buffer->input_bytes || !bytes ||
        offset % CEMU_FDM_KV_RECEIVE_UNIT_BYTES ||
        bytes % CEMU_FDM_KV_RECEIVE_UNIT_BYTES ||
        offset > wire_bytes || bytes > wire_bytes - offset) {
        return FDM_KV_RECEIVE_INVALID;
    }
    if (buffer->draining || buffer->sealed) {
        return FDM_KV_RECEIVE_INVALID;
    }

    //选空闲槽位
    for (uint32_t index = 0; index < buffer->input_slots; index++) {
        FDMKV_InputSlot_t *slot = &buffer->slots[index];
        if (slot->state == FDM_KV_SLOT_EMPTY) {
            if (empty == UINT32_MAX) {
                empty = index;
            }
        } else {
            if (slot->transfer_id == transfer_id) {
                chosen = index;
            }
            admission_busy |= slot->state == FDM_KV_SLOT_RECEIVING ||
                              slot->state == FDM_KV_SLOT_RECEIVED;
            if (slot->receive->header_valid &&
                slot->receive->header.operation == FDM_KV_OP_DRAIN) {
                return FDM_KV_RECEIVE_INVALID;
            }
        }
    }
    if (chosen == UINT32_MAX) {
        if (transfer_id <= buffer->last_transfer_id ||
            buffer->next_receive_id == UINT64_MAX) {
            return FDM_KV_RECEIVE_INVALID;
        }
        if (admission_busy) {
            return FDM_KV_RECEIVE_INVALID;
        }
        if (empty == UINT32_MAX) {
            return FDM_KV_RECEIVE_BUSY;
        }
        chosen = empty;
        FDMKV_InputSlot_t *slot = &buffer->slots[chosen];
        slot->transfer_id = transfer_id;
        slot->wire_bytes = wire_bytes;
        slot->receive->next_request_id = buffer->next_receive_id;
        slot->state = FDM_KV_SLOT_RECEIVING;
        buffer->last_transfer_id = transfer_id;
    }
    FDMKV_InputSlot_t *slot = &buffer->slots[chosen];
    if (slot->wire_bytes != wire_bytes) {
        return FDM_KV_RECEIVE_INVALID;
    }
    if (slot->state != FDM_KV_SLOT_RECEIVING || slot->dma_active) {
        return FDM_KV_RECEIVE_BUSY;
    }

    FDMKV_Buffer_t view = slot_view(buffer, chosen);
    FDMKV_ReceiveResult result = fdm_kv_receive_check(&view, offset, bytes);
    if (result != FDM_KV_RECEIVE_ACCEPTED) {
        return result;
    }

    slot->dma_offset = offset;
    slot->dma_bytes = bytes;
    slot->dma_active = 1;
    *slot_index = chosen;
    *destination = slot->input + offset;
    return FDM_KV_RECEIVE_ACCEPTED;
}

static bool receive_prefix(FDMKV_Buffer_t *buffer, const FDMKV_AppendHeader_t *header)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint32_t batches = header->version == CEMU_FDM_KV_LAYER_PACKET_VERSION ?
                       header->batch_count : 1;
    if (header->operation == FDM_KV_OP_DRAIN) {
        return true;
    }
    for (uint32_t batch = header->batch; batch < header->batch + batches; batch++) {
        uint64_t first = ((uint64_t)batch * geometry->layers + header->layer) *
                         geometry->heads + header->head_first;
        for (uint32_t head = 0; head < header->head_count; head++) {
            if (buffer->received_heads[first + head] != header->token_first) {
                return false;
            }
        }
    }
    for (uint32_t batch = header->batch; batch < header->batch + batches; batch++) {
        uint64_t first = ((uint64_t)batch * geometry->layers + header->layer) *
                         geometry->heads + header->head_first;
        for (uint32_t head = 0; head < header->head_count; head++) {
            buffer->received_heads[first + head] = header->token_first + header->token_count;
        }
    }
    return true;
}

FDMKV_ReceiveResult fdm_kv_slot_record(FDMKV_Buffer_t *buffer, uint32_t slot_index,
                                    uint64_t offset, uint64_t bytes, uint64_t ready_ns)
{
    if (!slot_buffer_valid(buffer) || slot_index >= buffer->input_slots) {
        return FDM_KV_RECEIVE_INVALID;
    }
    FDMKV_InputSlot_t *slot = &buffer->slots[slot_index];
    if (slot->state != FDM_KV_SLOT_RECEIVING || !slot->dma_active ||
        slot->dma_offset != offset || slot->dma_bytes != bytes) {
        return FDM_KV_RECEIVE_INVALID;
    }
    slot->dma_active = 0;
    FDMKV_Buffer_t view = slot_view(buffer, slot_index);
    FDMKV_ReceiveResult result = fdm_kv_receive_record(&view, offset, bytes, ready_ns);
    if ((result != FDM_KV_RECEIVE_ACCEPTED && result != FDM_KV_RECEIVE_READY) ||
        (slot->receive->header_valid && slot->receive->header.wire_bytes != slot->wire_bytes) ||
        (result == FDM_KV_RECEIVE_READY && !receive_prefix(buffer, &slot->receive->header))) {
        fdm_kv_slot_fail(buffer, slot_index);
        return FDM_KV_RECEIVE_INVALID;
    }
    if (result == FDM_KV_RECEIVE_READY) {
        slot->state = FDM_KV_SLOT_RECEIVED;
        buffer->next_receive_id++;
    }
    return result;
}

bool fdm_kv_slot_publish(FDMKV_Buffer_t *buffer, uint32_t slot_index, uint64_t now_ns)
{
    if (!slot_buffer_valid(buffer) || slot_index >= buffer->input_slots) {
        return false;
    }
    FDMKV_InputSlot_t *slot = &buffer->slots[slot_index];
    if (slot->state != FDM_KV_SLOT_RECEIVED || slot->dma_active ||
        slot->receive->state != FDM_KV_RX_READY ||
        now_ns < slot->receive->max_write_ready_ns) {
        return false;
    }
    slot->state = FDM_KV_SLOT_READY;
    return true;
}

bool fdm_kv_slot_begin(FDMKV_Buffer_t *buffer, uint32_t slot_index)
{
    if (!slot_buffer_valid(buffer) || slot_index >= buffer->input_slots) {
        return false;
    }
    FDMKV_InputSlot_t *slot = &buffer->slots[slot_index];
    if (slot->state != FDM_KV_SLOT_READY || slot->dma_active ||
        slot->receive->header.request_id != buffer->next_aggregate_id) {
        return false;
    }
    for (uint32_t index = 0; index < buffer->input_slots; index++) {
        if (buffer->slots[index].state == FDM_KV_SLOT_CONSUMING) {
            return false;
        }
    }
    slot->state = FDM_KV_SLOT_CONSUMING;
    buffer->active_slot = slot_index;
    buffer->input = slot->input;
    buffer->receive = slot->receive;
    return true;
}

bool fdm_kv_slot_release(FDMKV_Buffer_t *buffer, uint32_t slot_index)
{
    if (!slot_buffer_valid(buffer) || slot_index >= buffer->input_slots) {
        return false;
    }
    FDMKV_InputSlot_t *slot = &buffer->slots[slot_index];
    if (slot->state != FDM_KV_SLOT_CONSUMING || slot->dma_active ||
        slot->receive->header.request_id != buffer->next_aggregate_id) {
        return false;
    }
    FDMKV_Buffer_t view = slot_view(buffer, slot_index);
    if (!fdm_kv_receive_release(&view)) {
        return false;
    }
    slot->transfer_id = 0;
    slot->wire_bytes = 0;
    slot->dma_offset = 0;
    slot->dma_bytes = 0;
    slot->state = FDM_KV_SLOT_EMPTY;
    buffer->next_aggregate_id++;
    buffer->active_slot = 0;
    buffer->input = buffer->slots[0].input;
    buffer->receive = buffer->slots[0].receive;
    return true;
}
