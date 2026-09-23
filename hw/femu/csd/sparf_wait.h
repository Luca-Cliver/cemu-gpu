#ifndef CEMU_SPARF_WAIT_H
#define CEMU_SPARF_WAIT_H

#include <stdbool.h>
#include <stdint.h>
#include <glib.h>

static bool sparf_wait_target(uint64_t deadline, int64_t (*now)(void),
                              const gint *cancelled)
{
    for (;;) {
        if (cancelled && g_atomic_int_get(cancelled)) return false;
        if ((uint64_t)now() >= deadline) return true;
        g_usleep(10);
    }
}

#endif
