#include "../nvme.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int init_file_backend(SsdBackend *b, const char *path)
{
    struct stat stat_buffer;

    if (!path || !path[0]) {
        femu_err("File backend path is empty!\n");
        return -1;
    }

    b->fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (b->fd < 0) {
        femu_err("Failed to open SSD backend file %s: %s\n",
                 path, strerror(errno));
        return -1;
    }

    if (fstat(b->fd, &stat_buffer) < 0) {
        femu_err("Failed to stat SSD backend file %s: %s\n",
                 path, strerror(errno));
        close(b->fd);
        b->fd = -1;
        return -1;
    }

    if (stat_buffer.st_size < b->size && ftruncate(b->fd, b->size) < 0) {
        femu_err("Failed to resize SSD backend file %s to %" PRId64
                 " bytes: %s\n", path, b->size, strerror(errno));
        close(b->fd);
        b->fd = -1;
        return -1;
    }

    b->logical_space = mmap(NULL, b->size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, b->fd, 0);
    if (b->logical_space == MAP_FAILED) {
        femu_err("Failed to map SSD backend file %s: %s\n",
                 path, strerror(errno));
        b->logical_space = NULL;
        close(b->fd);
        b->fd = -1;
        return -1;
    }

    b->cuda_mirror = NULL;
    b->cuda_sync = false;
    backend_cuda_sync_init(b);
    femu_log("Using file-backed NVM: %s, size=%" PRId64 " bytes\n",
             path, b->size);
    return 0;
}

void free_file_backend(SsdBackend *b)
{
    backend_cuda_sync_fini(b);
    if (b->logical_space) {
        munmap(b->logical_space, b->size);
        b->logical_space = NULL;
    }
    if (b->fd >= 0) {
        close(b->fd);
        b->fd = -1;
    }
}
