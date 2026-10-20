#include "io_transport.h"
#include "cemu_client.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

using namespace cemu::instattention;

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

static void test_request_handles()
{
    auto data = std::make_shared<std::vector<std::uint8_t>>(4096, 17);
    std::promise<NvmeCompletion> pending, completed, failed;
    UringNvmeRequest first(pending.get_future().share(), data);
    UringNvmeRequest second(completed.get_future().share(), data);
    UringNvmeRequest third(failed.get_future().share(), data);
    require(!first.done(), "pending request reported ready");
    completed.set_value({12, 5});
    require(uring_wait_any({first, second}) == 1, "wait-any blocked behind pending request");
    require(second.result().status == 12 && second.result().result == 5,
            "positive NVMe status or repeated result lost");
    require(second.data().front() == 17, "request payload lifetime lost");
    failed.set_value({-EIO, 0});
    bool rejected = false;
    try { third.result(); }
    catch (const std::system_error &error) { rejected = error.code().value() == EIO; }
    require(rejected && third.done(), "failed CQE did not finish request");
    rejected = false;
    try { uring_wait_any({}); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "empty wait-any accepted");
}

static void test_cemu(const char *program)
{
    int descriptor = open("/dev/ng0n1", O_RDWR | O_CLOEXEC);
    require(descriptor >= 0, "missing native namespace /dev/ng0n1");
    try {
        std::array<unsigned char, 4096> data{};
        nvme_passthru_cmd command{};
        command.opcode = 0xc2;
        command.nsid = 1;
        command.cdw10 = 1;
        command.addr = reinterpret_cast<std::uint64_t>(data.data());
        command.data_len = data.size();
        require(ioctl(descriptor, NVME_IOCTL_IO_CMD, &command) == 0, "native INFO ioctl failed");
        auto expected = data;
        for (int iteration = 0; iteration < 128; ++iteration) {
            data.fill(0);
            require(uring_execute(descriptor, command) == 0, "native INFO uring failed");
            require(data == expected, "native INFO payload differs between transports");
        }
        require(data[0] == 'N' && data[1] == 'K' && data[2] == 'V' && data[3] == '1',
                "not a native KV namespace");
        for (int window = 0; window < 16; ++window) {
            std::vector<UringNvmeRequest> requests;
            for (int slot = 0; slot < 8; ++slot)
                requests.push_back(uring_submit_nvme(descriptor, command,
                                                     std::vector<std::uint8_t>(data.size())));
            while (!requests.empty()) {
                auto ready = uring_wait_any(requests);
                require(requests[ready].result().status == 0, "async native INFO failed");
                require(requests[ready].data() == std::vector<std::uint8_t>(expected.begin(), expected.end()),
                        "async native INFO completion/payload mismatch");
                requests.erase(requests.begin() + ready);
            }
        }
        command.cdw10 = 0;
        int expected_status = ioctl(descriptor, NVME_IOCTL_IO_CMD, &command);
        require(expected_status > 0 && uring_execute(descriptor, command) == expected_status,
                "positive NVMe status lost");
        close(descriptor);
    } catch (...) {
        close(descriptor);
        throw;
    }

    char directory[] = "/mnt/fdm0/uring-client-XXXXXX";
    require(mkdtemp(directory) != nullptr, "cannot create FDM test directory");
    std::string input_path = std::string(directory) + "/input";
    std::string output_path = std::string(directory) + "/output";
    try {
        for (const char *backend : {"ioctl", "uring"}) {
            setenv("CEMU_IO_BACKEND", backend, 1);
            CemuClient client;
            client.load_program(std::string("uring_test_") + backend, program, "vadd");
            client.activate_program();
            client.create_memory_ranges({{input_path, 0, 8192}, {output_path, 0, 4096}});
            std::array<int, 2048> input{};
            for (int index = 0; index < 1024; ++index) {
                input[index * 2] = index;
                input[index * 2 + 1] = index * 2;
            }
            client.write_range(0, input.data(), sizeof(input));
            ExecuteOptions options;
            options.cparam1 = 1024;
            options.runtime = 100000;
            std::array<unsigned char, 32> metadata{};
            require(client.execute(options, metadata.data(), metadata.size()) == 1024,
                    "Execute result lost");
            auto output = client.read_range(1, 4096);
            for (int index = 0; index < 1024; ++index) {
                int actual;
                std::memcpy(&actual, output.data() + index * sizeof(int), sizeof(int));
                require(actual == index * 3, "FDM/Execute result mismatch");
            }
        }
    } catch (...) {
        unlink(input_path.c_str());
        unlink(output_path.c_str());
        rmdir(directory);
        throw;
    }
    unlink(input_path.c_str());
    unlink(output_path.c_str());
    rmdir(directory);
    std::cout << "PASS: native INFO/status and ioctl-vs-uring FDM/MRS/Execute with metadata and runtime\n";
}

int main(int argc, char **argv)
{
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--cemu") != 0)) {
        std::cerr << "usage: test_io_transport [--cemu HOST_VADD_LIBRARY]\n";
        return 1;
    }
    test_request_handles();
    if (!io_uring_available()) {
        std::cout << "SKIP: io_uring transport not compiled\n";
        return 77;
    }
    char path[] = "/tmp/cemu-uring-test-XXXXXX";
    int descriptor = mkstemp(path);
    if (descriptor < 0) return 1;
    unlink(path);
    try {
        std::array<unsigned char, 4096> probe{};
        try {
            uring_read(descriptor, probe.data(), probe.size(), 0);
        } catch (const std::system_error &error) {
            if (std::string(error.what()).find("io_uring_queue_init") == std::string::npos) throw;
            if (error.code().value() != EPERM && error.code().value() != EINVAL &&
                error.code().value() != ENOSYS) throw;
            close(descriptor);
            std::cout << "SKIP: kernel/sandbox does not support SQE128/CQE32: " << error.what() << '\n';
            return 77;
        }
        std::vector<std::future<void>> tasks;
        for (unsigned int worker = 0; worker < 128; ++worker) {
            tasks.push_back(std::async(std::launch::async, [descriptor, worker] {
                std::array<unsigned char, 4096> input, output;
                input.fill(static_cast<unsigned char>(worker));
                for (int iteration = 0; iteration < 4; ++iteration) {
                    require(uring_write(descriptor, input.data(), input.size(), worker * 4096) == 4096,
                            "short write");
                    require(uring_read(descriptor, output.data(), output.size(), worker * 4096) == 4096,
                            "short read");
                    require(input == output, "completion routed to wrong buffer");
                }
            }));
        }
        for (auto &task : tasks) task.get();
        require(uring_read(descriptor, probe.data(), probe.size(), 127 * 4096 + 4000) == 96,
                "short read result lost");
        require(uring_read(descriptor, probe.data(), probe.size(), 128 * 4096) == 0,
                "EOF result lost");
        nvme_passthru_cmd command{};
        command.nsid = 3;
        command.opcode = 1;
        command.metadata = probe.size();
        command.data_len = probe.size();
        command.addr = reinterpret_cast<std::uint64_t>(probe.data());
        bool rejected = false;
        try { uring_execute(descriptor, command); }
        catch (const std::system_error &) { rejected = true; }
        require(rejected, "NVMe command accepted on regular file");
        std::vector<UringNvmeRequest> requests;
        int duplicate = dup(descriptor);
        require(duplicate >= 0, "cannot duplicate test fd");
        try {
            for (int slot = 0; slot < 8; ++slot)
                requests.push_back(uring_submit_nvme(duplicate, command,
                                                     std::vector<std::uint8_t>(probe.size())));
        } catch (...) {
            close(duplicate);
            throw;
        }
        close(duplicate);
        while (!requests.empty()) {
            auto ready = uring_wait_any(requests);
            rejected = false;
            try { requests[ready].result(); }
            catch (const std::system_error &) { rejected = true; }
            require(rejected, "async NVMe command accepted on regular file");
            requests.erase(requests.begin() + ready);
        }
        require(uring_read(descriptor, probe.data(), probe.size(), 0) == 4096,
                "one failed CQE poisoned dispatcher");
        close(descriptor);
        std::cout << "PASS: shared ring, 128 callers, async handles, wait-any, data, short reads, EOF and failed CQE isolation\n";
        descriptor = -1;
        if (argc == 3) test_cemu(argv[2]);
    } catch (const std::exception &error) {
        if (descriptor >= 0) close(descriptor);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
