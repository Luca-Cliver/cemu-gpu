#include "cemu_client.h"
#include "../../cemu_ioctl.h"
#include "../../../../hw/femu/fdm_kv_buffer_abi.h"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <utility>

using namespace cemu::instattention;

static unsigned pinned_creates, legacy_creates, binds, deletes;
static bool missing_kernel, bind_failure, delete_failure;

static void require(bool valid, const char *message)
{
    if (!valid) throw std::runtime_error(message);
}

extern "C" int __wrap_ioctl(int, unsigned long command, ...)
{
    va_list arguments;
    va_start(arguments, command);
    void *payload = va_arg(arguments, void *);
    va_end(arguments);
    if (command == IOCTL_CEMU_CREATE_MRS_PINNED || command == IOCTL_CEMU_CREATE_MRS) {
        if (command == IOCTL_CEMU_CREATE_MRS_PINNED) {
            pinned_creates++;
            if (missing_kernel) {
                errno = ENOTTY;
                return -1;
            }
        } else {
            legacy_creates++;
        }
        auto *request = static_cast<ioctl_create_mrs *>(payload);
        require(request->nr_fd == 2, "wrong MR count");
        request->rsid = 11;
        return 0;
    }
    if (command == IOCTL_CEMU_DOWNLOAD) {
        static_cast<ioctl_download *>(payload)->pind = 9;
        return 1;
    }
    if (command == IOCTL_CEMU_BIND_FDM_KV) {
        binds++;
        auto *request = static_cast<ioctl_fdm_kv_bind *>(payload);
        require(request->rsid == 11 && request->pind == 9 && request->bytes == 88, "wrong joint bind");
        FDMKV_BindSlotsParams_t descriptor;
        std::memcpy(&descriptor, reinterpret_cast<void *>(request->addr), sizeof(descriptor));
        require(descriptor.base.version == 2 && descriptor.input_slots == 2 &&
                descriptor.transfer_protocol == 1, "wrong slot protocol");
        if (bind_failure) {
            errno = EINVAL;
            return -1;
        }
        return 0;
    }
    if (command == IOCTL_CEMU_DELETE_MRS) {
        deletes++;
        require(static_cast<ioctl_create_mrs *>(payload)->rsid == 11, "wrong MRS deletion");
        if (delete_failure) {
            errno = EBUSY;
            return -1;
        }
        return 0;
    }
    if (command == IOCTL_CEMU_ACTIVATE || command == IOCTL_CEMU_DEACTIVATE || command == IOCTL_CEMU_UNLOAD)
        return 0;
    throw std::runtime_error("unexpected ioctl or unsafe passthrough fallback");
}

int main()
{
    char first[] = "/tmp/cemu-slot-meta-XXXXXX";
    char second[] = "/tmp/cemu-slot-data-XXXXXX";
    int meta = mkstemp(first), data = mkstemp(second);
    if (meta < 0 || data < 0) return 1;
    close(meta);
    close(data);
    int result = 0;
    try {
        const std::vector<MemoryRange> ranges = {{first, 0, 4096}, {second, 0, 8192}};
        FDMKV_BindSlotsParams_t wire = {};
        wire.base.version = 2;
        wire.input_slots = 2;
        wire.transfer_protocol = 1;
        {
            CemuClient client("/dev/null", "/dev/null");
            client.create_memory_ranges(ranges, true);
            client.load_program("slots", "/unused.so", "fdm_kv_aggregate");
            client.activate_program();
            client.bind_fdm_kv_buffer(&wire, sizeof(wire));
            CemuClient moved(std::move(client));
            delete_failure = true;
            bool rejected = false;
            try { moved.release_memory_ranges(); } catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "busy deletion was accepted");
            delete_failure = false;
            moved.release_memory_ranges();
        }
        require(pinned_creates == 1 && binds == 1 && deletes == 2, "pinned lifetime lost on move/retry");
        {
            CemuClient client("/dev/null", "/dev/null");
            client.create_memory_ranges(ranges, true);
            client.load_program("fail", "/unused.so", "fdm_kv_aggregate");
            client.activate_program();
            bind_failure = true;
            bool rejected = false;
            try { client.bind_fdm_kv_buffer(&wire, sizeof(wire)); } catch (const std::runtime_error &) { rejected = true; }
            bind_failure = false;
            require(rejected, "bind failure ignored");
            client.release_memory_ranges();
        }
        missing_kernel = true;
        {
            CemuClient client("/dev/null", "/dev/null");
            bool rejected = false;
            try { client.create_memory_ranges(ranges, true); } catch (const std::runtime_error &) { rejected = true; }
            require(rejected && !legacy_creates, "unsupported kernel silently fell back");
        }
        missing_kernel = false;
        {
            CemuClient client("/dev/null", "/dev/null");
            client.create_memory_ranges(ranges);
            client.load_program("legacy", "/unused.so", "fdm_kv_aggregate");
            client.activate_program();
            bool rejected = false;
            try { client.bind_fdm_kv_buffer(&wire, sizeof(wire)); } catch (const std::invalid_argument &) { rejected = true; }
            require(rejected && binds == 2, "untracked MRS reached tagged bind");
        }
        std::puts("FDM slot client: pinned create, joint bind, move, rollback, busy close and no fallback PASS");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    unlink(first);
    unlink(second);
    return result;
}
