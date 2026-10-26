#ifndef CEMU_FDM_KV_WRITER_H
#define CEMU_FDM_KV_WRITER_H

#include "fdm_kv_binding.h"

bool fdm_kv_writer_create(FDMKV_Binding_t *binding);
void fdm_kv_writer_wake(FDMKV_Binding_t *binding);
bool fdm_kv_writer_can_progress(const FDMKV_Binding_t *binding);
bool fdm_kv_writer_drain(FDMKV_Binding_t *binding);
void fdm_kv_writer_request_stop(FDMKV_Binding_t *binding);
void fdm_kv_writer_join(FDMKV_Binding_t *binding);
void fdm_kv_writer_free(FDMKV_Binding_t *binding);
void fdm_kv_writers_stop(FemuCtrl *ctrl);

#endif
