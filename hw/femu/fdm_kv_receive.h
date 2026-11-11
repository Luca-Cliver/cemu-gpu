#ifndef CEMU_FDM_KV_RECEIVE_H
#define CEMU_FDM_KV_RECEIVE_H

#include <stdbool.h>
#include "fdm_kv_buffer_abi.h"

typedef enum FDMKV_ReceiveResult {
    FDM_KV_RECEIVE_ACCEPTED,
    FDM_KV_RECEIVE_READY,
    FDM_KV_RECEIVE_INVALID,
    FDM_KV_RECEIVE_BUSY,
} FDMKV_ReceiveResult;

FDMKV_ReceiveResult fdm_kv_receive_check(const FDMKV_Buffer_t *buffer,
                                       uint64_t offset, uint64_t bytes);
FDMKV_ReceiveResult fdm_kv_receive_record(FDMKV_Buffer_t *buffer,
                                        uint64_t offset, uint64_t bytes,
                                        uint64_t ready_ns);
void fdm_kv_receive_fail(FDMKV_Buffer_t *buffer);
bool fdm_kv_receive_release(FDMKV_Buffer_t *buffer);

FDMKV_ReceiveResult fdm_kv_slot_acquire(FDMKV_Buffer_t *buffer,
                                     uint64_t transfer_id, uint64_t wire_bytes,
                                     uint64_t offset, uint64_t bytes,
                                     uint32_t *slot_index, uint8_t **destination);
FDMKV_ReceiveResult fdm_kv_slot_record(FDMKV_Buffer_t *buffer, uint32_t slot_index,
                                    uint64_t offset, uint64_t bytes, uint64_t ready_ns);
bool fdm_kv_slot_publish(FDMKV_Buffer_t *buffer, uint32_t slot_index, uint64_t now_ns);
bool fdm_kv_slot_begin(FDMKV_Buffer_t *buffer, uint32_t slot_index);
bool fdm_kv_slot_release(FDMKV_Buffer_t *buffer, uint32_t slot_index);
void fdm_kv_slot_fail(FDMKV_Buffer_t *buffer, uint32_t slot_index);

#endif
