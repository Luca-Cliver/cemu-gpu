#ifndef CEMU_INSTATTENTION_GUEST_CEMU_CLIENT_H
#define CEMU_INSTATTENTION_GUEST_CEMU_CLIENT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cemu::instattention {

enum class ProgramTarget {
    Host,
    CudaDevicePointer,
};

struct MemoryRange {
    std::string path;
    std::uint64_t offset;
    std::uint64_t size;
};

struct ExecuteOptions {
    std::uint64_t cparam1 = 0;
    std::uint64_t cparam2 = 0;
    std::uint32_t group = 0;
    std::uint32_t chunk_nlb = 0;
    std::uint32_t runtime = 0;
};

struct IoProfile {
    std::uint64_t prepare_ns = 0;
    std::uint64_t wait_ns = 0;
    std::uint64_t finish_ns = 0;
    std::uint64_t bytes = 0;
};

class CemuClient {
public:
    CemuClient(std::string control_path = "/dev/nvme0c3",
               std::string namespace_path = "/dev/ng0n3");
    ~CemuClient();

    CemuClient(const CemuClient &) = delete;
    CemuClient &operator=(const CemuClient &) = delete;
    CemuClient(CemuClient &&other) noexcept;
    CemuClient &operator=(CemuClient &&other) noexcept;

    std::uint16_t load_program(const std::string &program_name,
                               const std::string &shared_library_path,
                               const std::string &function_name,
                               ProgramTarget target = ProgramTarget::Host,
                               std::uint32_t runtime = 0,
                               std::uint16_t runtime_scale_tenths = 0,
                               bool indirect = false,
                               bool replace_existing = false);
    void activate_program();

    std::uint16_t create_memory_ranges(const std::vector<MemoryRange> &ranges, bool pinned = false);
    void release_memory_ranges();
    void bind_fdm_kv_buffer(const void *metadata, std::size_t metadata_size);

    void write_range(std::size_t range_index,
                     const void *data,
                     std::size_t size,
                     std::uint64_t range_offset = 0,
                     IoProfile *profile = nullptr);
    std::vector<std::uint8_t> read_range(std::size_t range_index,
                                         std::size_t size,
                                         std::uint64_t range_offset = 0,
                                         IoProfile *profile = nullptr) const;
    void read_range_into(std::size_t range_index,
                         void *data,
                         std::size_t size,
                         std::uint64_t range_offset = 0,
                         IoProfile *profile = nullptr) const;

    std::uint32_t execute(const ExecuteOptions &options,
                          const void *metadata = nullptr,
                          std::size_t metadata_size = 0,
                          IoProfile *profile = nullptr);

    void close() noexcept;

    std::uint16_t program_id() const noexcept;
    std::uint16_t memory_range_set_id() const noexcept;
    std::size_t memory_range_count() const noexcept;

private:
    void ensure_open() const;
    void ensure_program_loaded() const;
    void ensure_program_active() const;
    void ensure_memory_ranges_created() const;
    void validate_range_access(std::size_t range_index,
                               std::size_t size,
                               std::uint64_t range_offset) const;

    int control_fd_ = -1;
    int namespace_fd_ = -1;
    std::string program_name_;
    std::int32_t program_type_ = -1;
    std::int32_t program_target_ = -1;
    std::uint16_t program_id_ = 0;
    std::uint16_t memory_range_set_id_ = 0;
    bool program_loaded_ = false;
    bool program_active_ = false;
    bool memory_ranges_created_ = false;
    bool memory_ranges_pinned_ = false;
    std::vector<MemoryRange> memory_ranges_;
    std::vector<int> memory_fds_;
};

} // namespace cemu::instattention

#endif
