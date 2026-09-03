#include "cemu_client.h"

#include "../../cemu_ioctl.h"
#include "../../util.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace cemu::instattention {
namespace {

constexpr std::size_t kIoAlignment = 4096;
constexpr std::size_t kBlockAlignment = 512;

[[noreturn]] void throw_system_error(const std::string &operation)
{
    const int error = errno;
    throw std::runtime_error(operation + ": " + std::strerror(error));
}

std::size_t round_up(std::size_t value, std::size_t alignment)
{
    if (value == 0) {
        return 0;
    }
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("aligned size overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t size) : size_(size)
    {
        if (size_ == 0) {
            return;
        }
        const int result = posix_memalign(&data_, kIoAlignment, size_);
        if (result != 0) {
            throw std::runtime_error("posix_memalign: " +
                                     std::string(std::strerror(result)));
        }
        std::memset(data_, 0, size_);
    }

    ~AlignedBuffer()
    {
        std::free(data_);
    }

    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;

    void *data() noexcept
    {
        return data_;
    }

    const void *data() const noexcept
    {
        return data_;
    }

private:
    void *data_ = nullptr;
    std::size_t size_ = 0;
};

void pwrite_all(int fd, const void *data, std::size_t size, off_t offset)
{
    const auto *cursor = static_cast<const std::uint8_t *>(data);
    std::size_t written = 0;
    while (written < size) {
        const ssize_t result = pwrite(fd, cursor + written, size - written,
                                      offset + static_cast<off_t>(written));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw_system_error("pwrite FDM range");
        }
        if (result == 0) {
            throw std::runtime_error("pwrite FDM range returned zero");
        }
        written += static_cast<std::size_t>(result);
    }
}

void pread_all(int fd, void *data, std::size_t size, off_t offset)
{
    auto *cursor = static_cast<std::uint8_t *>(data);
    std::size_t read_size = 0;
    while (read_size < size) {
        const ssize_t result = pread(fd, cursor + read_size, size - read_size,
                                     offset + static_cast<off_t>(read_size));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw_system_error("pread FDM range");
        }
        if (result == 0) {
            throw std::runtime_error("unexpected end of FDM range");
        }
        read_size += static_cast<std::size_t>(result);
    }
}

void close_fd(int &fd) noexcept
{
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

} // namespace

CemuClient::CemuClient(std::string control_path, std::string namespace_path)
{
    control_fd_ = ::open(control_path.c_str(), O_RDWR);
    if (control_fd_ < 0) {
        throw_system_error("open " + control_path);
    }

    namespace_fd_ = ::open(namespace_path.c_str(), O_RDWR);
    if (namespace_fd_ < 0) {
        const int error = errno;
        close_fd(control_fd_);
        errno = error;
        throw_system_error("open " + namespace_path);
    }
}

CemuClient::~CemuClient()
{
    close();
}

CemuClient::CemuClient(CemuClient &&other) noexcept
    : control_fd_(std::exchange(other.control_fd_, -1)),
      namespace_fd_(std::exchange(other.namespace_fd_, -1)),
      program_name_(std::move(other.program_name_)),
      program_type_(std::exchange(other.program_type_, -1)),
      program_target_(std::exchange(other.program_target_, -1)),
      program_id_(std::exchange(other.program_id_, 0)),
      memory_range_set_id_(std::exchange(other.memory_range_set_id_, 0)),
      program_loaded_(std::exchange(other.program_loaded_, false)),
      program_active_(std::exchange(other.program_active_, false)),
      memory_ranges_created_(std::exchange(other.memory_ranges_created_, false)),
      memory_ranges_(std::move(other.memory_ranges_)),
      memory_fds_(std::move(other.memory_fds_))
{
    other.memory_fds_.clear();
}

CemuClient &CemuClient::operator=(CemuClient &&other) noexcept
{
    if (this == &other) {
        return *this;
    }

    close();
    control_fd_ = std::exchange(other.control_fd_, -1);
    namespace_fd_ = std::exchange(other.namespace_fd_, -1);
    program_name_ = std::move(other.program_name_);
    program_type_ = std::exchange(other.program_type_, -1);
    program_target_ = std::exchange(other.program_target_, -1);
    program_id_ = std::exchange(other.program_id_, 0);
    memory_range_set_id_ = std::exchange(other.memory_range_set_id_, 0);
    program_loaded_ = std::exchange(other.program_loaded_, false);
    program_active_ = std::exchange(other.program_active_, false);
    memory_ranges_created_ = std::exchange(other.memory_ranges_created_, false);
    memory_ranges_ = std::move(other.memory_ranges_);
    memory_fds_ = std::move(other.memory_fds_);
    other.memory_fds_.clear();
    return *this;
}

std::uint16_t CemuClient::load_program(const std::string &program_name,
                                       const std::string &shared_library_path,
                                       const std::string &function_name,
                                       ProgramTarget target,
                                       std::uint32_t runtime,
                                       std::uint16_t runtime_scale_tenths,
                                       bool indirect,
                                       bool replace_existing)
{
    ensure_open();
    if (program_loaded_) {
        throw std::logic_error("a CEMU program is already loaded");
    }
    if (program_name.empty() || shared_library_path.empty() || function_name.empty()) {
        throw std::invalid_argument("program name, shared-library path and function name are required");
    }

    program_name_ = program_name;
    ioctl_download download = {};
    download.name = program_name_.c_str();
    if (target == ProgramTarget::CudaDevicePointer) {
        prep_cuda_lib(shared_library_path.c_str(), function_name.c_str(), &download);
    } else {
        prep_shared_library(shared_library_path.c_str(), function_name.c_str(), &download);
    }
    std::unique_ptr<void, decltype(&std::free)> payload(download.addr, &std::free);
    download.runtime = static_cast<std::int32_t>(runtime);
    download.runtime_scale = static_cast<std::int32_t>(runtime_scale_tenths);
    download.indirect = indirect ? 1 : 0;

    int result = ioctl(control_fd_, IOCTL_CEMU_DOWNLOAD, &download);
    if (result <= 0 && errno == EEXIST && replace_existing) {
        ioctl(control_fd_, IOCTL_CEMU_DEACTIVATE, &download);
        if (ioctl(control_fd_, IOCTL_CEMU_UNLOAD, &download) != 0) {
            throw_system_error("unload existing CEMU program");
        }
        download.pind = 0;
        result = ioctl(control_fd_, IOCTL_CEMU_DOWNLOAD, &download);
    }
    if (result <= 0) {
        throw_system_error("download CEMU program " + program_name_);
    }

    program_type_ = download.ptype;
    program_target_ = download.target;
    program_id_ = static_cast<std::uint16_t>(download.pind);
    program_loaded_ = true;
    return program_id_;
}

void CemuClient::activate_program()
{
    ensure_program_loaded();
    if (program_active_) {
        return;
    }

    ioctl_download download = {};
    download.name = program_name_.c_str();
    download.ptype = program_type_;
    download.target = program_target_;
    download.pind = program_id_;
    if (ioctl(control_fd_, IOCTL_CEMU_ACTIVATE, &download) != 0) {
        throw_system_error("activate CEMU program " + program_name_);
    }
    program_active_ = true;
}

std::uint16_t CemuClient::create_memory_ranges(const std::vector<MemoryRange> &ranges)
{
    ensure_open();
    if (memory_ranges_created_) {
        throw std::logic_error("a CEMU memory range set is already created");
    }
    if (ranges.empty() || ranges.size() >= 128) {
        throw std::invalid_argument("memory range count must be between 1 and 127");
    }

    std::vector<int> fds;
    fds.reserve(ranges.size());
    try {
        for (const MemoryRange &range : ranges) {
            if (range.path.empty() || range.size == 0) {
                throw std::invalid_argument("memory range path and size are required");
            }
            if (range.offset % kBlockAlignment != 0 ||
                range.size % kBlockAlignment != 0) {
                throw std::invalid_argument("memory range offset and size must be 512-byte aligned");
            }
            if (range.offset > std::numeric_limits<std::uint64_t>::max() - range.size ||
                range.offset + range.size > static_cast<std::uint64_t>(LLONG_MAX)) {
                throw std::overflow_error("memory range size overflow");
            }

            const int fd = ::open(range.path.c_str(), O_RDWR | O_CREAT, 0666);
            if (fd < 0) {
                throw_system_error("open " + range.path);
            }
            fds.push_back(fd);

            struct stat file_stat = {};
            if (fstat(fd, &file_stat) != 0) {
                throw_system_error("fstat " + range.path);
            }

            const std::uint64_t required_size = range.offset + range.size;
            if (file_stat.st_size == 0) {
                const std::size_t allocation_size = round_up(
                    static_cast<std::size_t>(required_size), kIoAlignment);
                const int result = posix_fallocate(fd, 0, static_cast<off_t>(allocation_size));
                if (result != 0) {
                    throw std::runtime_error("posix_fallocate " + range.path + ": " +
                                             std::string(std::strerror(result)));
                }
            } else if (static_cast<std::uint64_t>(file_stat.st_size) < required_size) {
                throw std::runtime_error(range.path + " is smaller than the requested memory range");
            }
        }

        std::vector<long long> offsets;
        std::vector<long long> sizes;
        offsets.reserve(ranges.size());
        sizes.reserve(ranges.size());
        for (const MemoryRange &range : ranges) {
            offsets.push_back(static_cast<long long>(range.offset));
            sizes.push_back(static_cast<long long>(range.size));
        }

        ioctl_create_mrs request = {};
        request.nr_fd = static_cast<int>(fds.size());
        request.fd = fds.data();
        request.off = offsets.data();
        request.size = sizes.data();
        if (ioctl(control_fd_, IOCTL_CEMU_CREATE_MRS, &request) != 0) {
            throw_system_error("create CEMU memory range set");
        }

        memory_ranges_ = ranges;
        memory_fds_ = std::move(fds);
        memory_range_set_id_ = request.rsid;
        memory_ranges_created_ = true;
        return memory_range_set_id_;
    } catch (...) {
        for (int fd : fds) {
            ::close(fd);
        }
        throw;
    }
}

void CemuClient::write_range(std::size_t range_index,
                             const void *data,
                             std::size_t size,
                             std::uint64_t range_offset)
{
    validate_range_access(range_index, size, range_offset);
    if (size == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("write_range data is null");
    }

    const std::size_t io_size = round_up(size, kBlockAlignment);
    if (range_offset + io_size > memory_ranges_[range_index].size) {
        throw std::out_of_range("aligned write exceeds the memory range");
    }

    AlignedBuffer buffer(io_size);
    std::memcpy(buffer.data(), data, size);
    const std::uint64_t file_offset = memory_ranges_[range_index].offset + range_offset;
    pwrite_all(memory_fds_[range_index], buffer.data(), io_size,
               static_cast<off_t>(file_offset));
}

std::vector<std::uint8_t> CemuClient::read_range(std::size_t range_index,
                                                 std::size_t size,
                                                 std::uint64_t range_offset) const
{
    validate_range_access(range_index, size, range_offset);
    if (size == 0) {
        return {};
    }

    const std::size_t io_size = round_up(size, kBlockAlignment);
    if (range_offset + io_size > memory_ranges_[range_index].size) {
        throw std::out_of_range("aligned read exceeds the memory range");
    }

    AlignedBuffer buffer(io_size);
    const std::uint64_t file_offset = memory_ranges_[range_index].offset + range_offset;
    pread_all(memory_fds_[range_index], buffer.data(), io_size,
              static_cast<off_t>(file_offset));

    std::vector<std::uint8_t> result(size);
    std::memcpy(result.data(), buffer.data(), size);
    return result;
}

std::uint32_t CemuClient::execute(const ExecuteOptions &options,
                                  const void *metadata,
                                  std::size_t metadata_size)
{
    ensure_program_active();
    ensure_memory_ranges_created();
    if (metadata_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CEMU execute metadata is too large");
    }
    if (metadata_size != 0 && metadata == nullptr) {
        throw std::invalid_argument("CEMU execute metadata is null");
    }

    AlignedBuffer metadata_buffer(metadata_size);
    if (metadata_size != 0) {
        std::memcpy(metadata_buffer.data(), metadata, metadata_size);
    }

    nvme_passthru_cmd command = {};
    prep_nvme_passthru_program_execute(
        &command,
        options.cparam1,
        options.cparam2,
        program_id_,
        memory_range_set_id_,
        0,
        options.group,
        options.chunk_nlb,
        options.runtime,
        metadata_size == 0 ? nullptr : metadata_buffer.data(),
        static_cast<std::uint32_t>(metadata_size));

    const int status = ioctl(namespace_fd_, NVME_IOCTL_IO_CMD, &command);
    if (status < 0) {
        throw_system_error("execute CEMU program " + program_name_);
    }
    if (status != 0) {
        throw std::runtime_error(
            "execute CEMU program " + program_name_ +
            ": NVMe status " + std::to_string(status));
    }
    return command.result;
}

void CemuClient::close() noexcept
{
    if (control_fd_ >= 0 && memory_ranges_created_) {
        ioctl_create_mrs request = {};
        request.rsid = memory_range_set_id_;
        ioctl(control_fd_, IOCTL_CEMU_DELETE_MRS, &request);
    }
    memory_ranges_created_ = false;
    memory_range_set_id_ = 0;

    for (int &fd : memory_fds_) {
        close_fd(fd);
    }
    memory_fds_.clear();
    memory_ranges_.clear();

    if (control_fd_ >= 0 && program_loaded_) {
        ioctl_download download = {};
        download.name = program_name_.c_str();
        download.ptype = program_type_;
        download.target = program_target_;
        download.pind = program_id_;
        if (program_active_) {
            ioctl(control_fd_, IOCTL_CEMU_DEACTIVATE, &download);
        }
        ioctl(control_fd_, IOCTL_CEMU_UNLOAD, &download);
    }
    program_active_ = false;
    program_loaded_ = false;
    program_id_ = 0;
    program_type_ = -1;
    program_target_ = -1;
    program_name_.clear();

    close_fd(namespace_fd_);
    close_fd(control_fd_);
}

std::uint16_t CemuClient::program_id() const noexcept
{
    return program_id_;
}

std::uint16_t CemuClient::memory_range_set_id() const noexcept
{
    return memory_range_set_id_;
}

std::size_t CemuClient::memory_range_count() const noexcept
{
    return memory_ranges_.size();
}

void CemuClient::ensure_open() const
{
    if (control_fd_ < 0 || namespace_fd_ < 0) {
        throw std::logic_error("CEMU client is closed");
    }
}

void CemuClient::ensure_program_loaded() const
{
    ensure_open();
    if (!program_loaded_) {
        throw std::logic_error("no CEMU program is loaded");
    }
}

void CemuClient::ensure_program_active() const
{
    ensure_program_loaded();
    if (!program_active_) {
        throw std::logic_error("the CEMU program is not active");
    }
}

void CemuClient::ensure_memory_ranges_created() const
{
    ensure_open();
    if (!memory_ranges_created_) {
        throw std::logic_error("no CEMU memory range set is created");
    }
}

void CemuClient::validate_range_access(std::size_t range_index,
                                       std::size_t size,
                                       std::uint64_t range_offset) const
{
    ensure_memory_ranges_created();
    if (range_index >= memory_ranges_.size()) {
        throw std::out_of_range("invalid CEMU memory range index");
    }
    if (range_offset % kBlockAlignment != 0) {
        throw std::invalid_argument("memory range access offset must be 512-byte aligned");
    }
    if (range_offset > memory_ranges_[range_index].size ||
        size > memory_ranges_[range_index].size - range_offset) {
        throw std::out_of_range("memory range access exceeds the configured range");
    }
}

} // namespace cemu::instattention
