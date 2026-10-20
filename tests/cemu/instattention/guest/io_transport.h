#ifndef CEMU_INSTATTENTION_IO_TRANSPORT_H
#define CEMU_INSTATTENTION_IO_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>
#include <linux/nvme_ioctl.h>
#include <sys/types.h>

namespace cemu::instattention {

struct NvmeCompletion {
    int status;
    std::uint64_t result;
};

class UringNvmeRequest {
public:
    UringNvmeRequest(std::shared_future<NvmeCompletion> completion,
                     std::shared_ptr<std::vector<std::uint8_t>> data);
    bool done() const;
    NvmeCompletion result() const;
    const std::vector<std::uint8_t> &data() const;

private:
    std::shared_future<NvmeCompletion> completion_;
    std::shared_ptr<std::vector<std::uint8_t>> data_;
};

UringNvmeRequest uring_submit_nvme(int descriptor, const nvme_passthru_cmd &command,
                                  std::vector<std::uint8_t> payload);
std::size_t uring_wait_any(const std::vector<UringNvmeRequest> &requests);

bool use_io_uring();
bool io_uring_available();
int uring_execute(int descriptor, nvme_passthru_cmd &command);
ssize_t uring_read(int descriptor, void *data, std::size_t size, std::uint64_t offset);
ssize_t uring_write(int descriptor, const void *data, std::size_t size, std::uint64_t offset);
ssize_t uring_copy(int source, int destination, std::size_t size,
                   std::uint64_t source_offset, std::uint64_t destination_offset);

}

#endif
