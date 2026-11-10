#include "cemu_client.h"
#include "../../cemu_ioctl.h"

#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <linux/nvme_ioctl.h>
#include <stdexcept>
#include <unistd.h>

using namespace cemu::instattention;

static int execute_status;
static nvme_passthru_cmd last_command;
static std::array<unsigned char, 43> last_metadata;
static void *last_read_destination;

extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void *data, size_t size, off_t offset)
{
    last_read_destination = data;
    return __real_pread(fd, data, size, offset);
}

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

extern "C" int __wrap_ioctl(int, unsigned long request, ...)
{
    va_list arguments;
    va_start(arguments, request);
    void *payload = va_arg(arguments, void *);
    va_end(arguments);
    if (request == IOCTL_CEMU_DOWNLOAD) {
        static_cast<ioctl_download *>(payload)->pind = 9;
        return 1;
    }
    if (request == IOCTL_CEMU_CREATE_MRS) {
        static_cast<ioctl_create_mrs *>(payload)->rsid = 7;
        return 0;
    }
    if (request == NVME_IOCTL_IO_CMD) {
        auto *command = static_cast<nvme_passthru_cmd *>(payload);
        last_command = *command;
        require(command->data_len == last_metadata.size(), "metadata size changed");
        std::memcpy(last_metadata.data(), reinterpret_cast<void *>(command->addr), last_metadata.size());
        command->result = 71;
        errno = EIO;
        return execute_status;
    }
    if (request == IOCTL_CEMU_ACTIVATE || request == IOCTL_CEMU_DEACTIVATE ||
            request == IOCTL_CEMU_UNLOAD || request == IOCTL_CEMU_DELETE_MRS) return 0;
    errno = ENOTTY;
    return -1;
}

int main()
{
    setenv("CEMU_IO_BACKEND", "ioctl", 1);
    char path[] = "/tmp/cemu-client-profile-XXXXXX";
    int descriptor = mkstemp(path);
    if (descriptor < 0) return 1;
    close(descriptor);
    try {
        CemuClient client("/dev/null", "/dev/null");
        client.load_program("profile_test", "/unused/host.so", "test");
        client.activate_program();
        client.create_memory_ranges({{path, 512, 4096}});
        std::array<unsigned char, 513> input;
        input.fill(37);
        std::array<unsigned char, 43> metadata;
        metadata.fill(19);
        nvme_passthru_cmd original{};
        for (bool enabled : {false, true}) {
            IoProfile timing;
            IoProfile *profile = enabled ? &timing : nullptr;
            client.write_range(0, input.data(), input.size(), 512, profile);
            if (enabled) require(timing.bytes == 1024, "write wire bytes not aligned");
            auto result = client.read_range(0, input.size(), 512, profile);
            require(result == std::vector<std::uint8_t>(input.begin(), input.end()), "profile changed IO data");
            if (enabled) require(timing.bytes == 1024, "read wire bytes not aligned");
            auto padded = client.read_range(0, 1024, 512);
            for (std::size_t index = input.size(); index < padded.size(); ++index)
                require(padded[index] == 0, "padding changed");
            alignas(4096) std::array<unsigned char, 4096> direct;
            direct.fill(99);
            client.read_range_into(0, direct.data(), 1024, 512, profile);
            require(last_read_destination == direct.data(), "aligned output was not read directly");
            require(std::memcmp(direct.data(), padded.data(), 1024) == 0, "direct read data mismatch");
            require(direct[1024] == 99, "direct read overran output");
            if (enabled) require(timing.bytes == 1024, "direct read profile mismatch");
            direct.fill(99);
            client.read_range_into(0, direct.data() + 1, input.size(), 512, profile);
            require(last_read_destination != direct.data() + 1, "unaligned output bypassed bounce buffer");
            require(std::memcmp(direct.data() + 1, input.data(), input.size()) == 0, "fallback read mismatch");
            require(direct[0] == 99 && direct[514] == 99, "padded read overran caller buffer");
            direct.fill(99);
            client.read_range_into(0, direct.data(), input.size(), 512, profile);
            require(last_read_destination != direct.data(), "partial sector read bypassed bounce buffer");
            require(direct[513] == 99, "partial sector read wrote padding into output");
            bool bad_range = false;
            try { client.read_range_into(0, direct.data(), 513, 3584); }
            catch (const std::out_of_range &) { bad_range = true; }
            require(bad_range, "read_into accepted out-of-bounds transfer");
            ExecuteOptions options;
            options.cparam1 = 123;
            options.cparam2 = 456;
            options.runtime = 999;
            require(client.execute(options, metadata.data(), metadata.size(), profile) == 71,
                    "profile changed EXEC result");
            require(last_metadata == metadata, "profile changed metadata");
            last_command.addr = 0;
            if (enabled) {
                require(timing.bytes == metadata.size(), "EXEC transfer count mismatch");
                require(std::memcmp(&original, &last_command, sizeof(original)) == 0, "EXEC command changed");
            } else {
                original = last_command;
            }
            for (int status : {-1, 5}) {
                execute_status = status;
                bool rejected = false;
                try { client.execute(options, metadata.data(), metadata.size(), profile); }
                catch (const std::runtime_error &error) {
                    const std::string expected = status < 0 ? std::strerror(EIO) : "NVMe status 5";
                    rejected = std::string(error.what()).find(expected) != std::string::npos;
                }
                require(rejected, "profile changed EXEC error");
            }
            execute_status = 0;
            timing.bytes = 1234;
            client.write_range(0, nullptr, 0, 0, &timing);
            require(timing.bytes == 0 && timing.wait_ns == 0, "empty IO retained old profile");
            client.read_range_into(0, nullptr, 0, 0, &timing);
            require(timing.bytes == 0 && timing.wait_ns == 0, "empty read retained old profile");
        }
        client.close();
        unlink(path);
        std::cout << "PASS: profiled/unprofiled client bytes, padding, offsets, EXEC command/result/errors; mocked management/EXEC\n";
        return 0;
    } catch (const std::exception &error) {
        unlink(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
