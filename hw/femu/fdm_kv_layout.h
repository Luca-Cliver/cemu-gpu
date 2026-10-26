#ifndef CEMU_FDM_KV_LAYOUT_H
#define CEMU_FDM_KV_LAYOUT_H

#include <stdbool.h>
#include "fdm_kv_buffer_abi.h"

bool fdm_kv_layout_init(
    FDMKV_Layout_t *layout,
    const FDMKV_Geometry_t *geometry
);

#endif