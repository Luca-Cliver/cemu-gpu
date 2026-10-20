#include "io_transport.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef CEMU_HAVE_URING
#include <liburing.h>
#include <algorithm>
#include <deque>
#include <fcntl.h>
#include <future>
#include <memory>
#include <poll.h>
#include <sys/eventfd.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#endif

namespace cemu::instattention {

namespace {
std::mutex completion_mutex;
std::condition_variable completion_ready;
}

UringNvmeRequest::UringNvmeRequest(std::shared_future<NvmeCompletion> completion,
                                  std::shared_ptr<std::vector<std::uint8_t>> data)
    : completion_(std::move(completion)), data_(std::move(data)) {}

bool UringNvmeRequest::done() const
{
    return completion_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

NvmeCompletion UringNvmeRequest::result() const
{
    auto completion = completion_.get();
    if (completion.status < 0)
        throw std::system_error(-completion.status, std::generic_category(), "io_uring completion");
    return completion;
}

const std::vector<std::uint8_t> &UringNvmeRequest::data() const { return *data_; }

std::size_t uring_wait_any(const std::vector<UringNvmeRequest> &requests)
{
    if (requests.empty()) throw std::invalid_argument("cannot wait on an empty request list");
    std::size_t ready = 0;
    std::unique_lock<std::mutex> guard(completion_mutex);
    completion_ready.wait(guard, [&] {
        for (ready = 0; ready < requests.size(); ++ready)
            if (requests[ready].done()) return true;
        return false;
    });
    return ready;
}

bool use_io_uring()
{
    const char *backend = std::getenv("CEMU_IO_BACKEND");
    if (!backend || std::strcmp(backend, "ioctl") == 0) return false;
    if (std::strcmp(backend, "uring") != 0)
        throw std::invalid_argument("CEMU_IO_BACKEND must be ioctl or uring");
    if (!io_uring_available())
        throw std::runtime_error("io_uring backend unavailable: rebuild with liburing >= 2.2 and NVMe uring headers");
    return true;
}

bool io_uring_available()
{
#ifdef CEMU_HAVE_URING
    return true;
#else
    return false;
#endif
}

#ifdef CEMU_HAVE_URING
namespace {

enum class Operation { Execute, Read, Write, Copy };

using Completion = NvmeCompletion;

struct Request {
    Operation operation;
    int descriptor = -1;
    int source = -1;
    void *data = nullptr;
    std::shared_ptr<std::vector<std::uint8_t>> owned_data;
    std::size_t size;
    std::uint64_t offset = 0;
    std::uint64_t source_offset = 0;
    nvme_uring_cmd command = {};
    bool submitted = false;
    std::promise<Completion> promise;

    void finish(Completion completion)
    {
        std::lock_guard<std::mutex> guard(completion_mutex);
        promise.set_value(completion);
        completion_ready.notify_all();
    }

    void fail(std::exception_ptr error)
    {
        std::lock_guard<std::mutex> guard(completion_mutex);
        promise.set_exception(error);
        completion_ready.notify_all();
    }

    Request(Operation operation, int descriptor, std::size_t size)
        : operation(operation), size(size)
    {
        if (size > INT_MAX) throw std::invalid_argument("io_uring transfer exceeds INT_MAX");
        this->descriptor = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (this->descriptor < 0) throw std::system_error(errno, std::generic_category(), "dup io_uring fd");
    }

    ~Request()
    {
        if (descriptor >= 0) close(descriptor);
        if (source >= 0) close(source);
    }
};

class Dispatcher {
public:
    Dispatcher()
    {
        int status = io_uring_queue_init(64, &ring_, IORING_SETUP_SQE128 | IORING_SETUP_CQE32);
        if (status < 0) throw std::system_error(-status, std::generic_category(), "io_uring_queue_init");
        wake_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (wake_ < 0) {
            int error = errno;
            io_uring_queue_exit(&ring_);
            throw std::system_error(error, std::generic_category(), "io_uring eventfd");
        }
        try {
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            close(wake_);
            io_uring_queue_exit(&ring_);
            throw;
        }
    }

    ~Dispatcher()
    {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
        }
        notify();
        worker_.join();
        close(wake_);
    }

    std::future<Completion> submit(const std::shared_ptr<Request> &request)
    {
        auto result = request->promise.get_future();
        {
            std::unique_lock<std::mutex> guard(mutex_);
            space_.wait(guard, [this] { return pending_.size() < 64 || failure_ || stopping_; });
            if (failure_) std::rethrow_exception(failure_);
            if (stopping_) throw std::runtime_error("io_uring dispatcher is closing");
            pending_.push_back(request);
        }
        notify();
        return result;
    }

    Completion execute(const std::shared_ptr<Request> &request)
    {
        Completion completion = submit(request).get();
        if (completion.status < 0)
            throw std::system_error(-completion.status, std::generic_category(), "io_uring completion");
        return completion;
    }

private:
    void notify()
    {
        std::uint64_t signal = 1;
        while (write(wake_, &signal, sizeof(signal)) < 0 && errno == EINTR) {}
    }

    void prepare(io_uring_sqe *entry, Request *request)
    {
        std::memset(entry, 0, 128);
        switch (request->operation) {
        case Operation::Execute:
            entry->opcode = IORING_OP_URING_CMD;
            entry->cmd_op = NVME_URING_CMD_IO;
            entry->fd = request->descriptor;
            std::memcpy(entry->cmd, &request->command, sizeof(request->command));
            break;
        case Operation::Read:
            io_uring_prep_read(entry, request->descriptor, request->data, request->size, request->offset);
            break;
        case Operation::Write:
            io_uring_prep_write(entry, request->descriptor, request->data, request->size, request->offset);
            break;
        case Operation::Copy:
            io_uring_prep_rw(55, entry, request->descriptor, nullptr, request->size, request->offset);
            entry->splice_fd_in = request->source;
            entry->splice_off_in = request->source_offset;
            break;
        }
        io_uring_sqe_set_data(entry, request);
    }

    void run() noexcept
    {
        std::unordered_map<Request *, std::shared_ptr<Request>> active;
        std::deque<Request *> submission_order;
        try {
            while (true) {
                bool submitted = false;
                {
                    std::lock_guard<std::mutex> guard(mutex_);
                    while (!pending_.empty() && active.size() < 64) {
                        auto *entry = io_uring_get_sqe(&ring_);
                        if (!entry) break;
                        auto request = pending_.front();
                        active.emplace(request.get(), request);
                        pending_.pop_front();
                        submission_order.push_back(request.get());
                        prepare(entry, request.get());
                        submitted = true;
                    }
                    if (stopping_ && pending_.empty() && active.empty()) break;
                }
                space_.notify_all();
                if (submitted) {
                    while (io_uring_sq_ready(&ring_)) {
                        int status = io_uring_submit(&ring_);
                        if (status == -EINTR) continue;
                        if (status < 0) throw std::system_error(-status, std::generic_category(), "io_uring_submit");
                        if (!status) throw std::runtime_error("io_uring submission made no progress");
                        while (status-- > 0) {
                            submission_order.front()->submitted = true;
                            submission_order.pop_front();
                        }
                    }
                }
                bool completed = false;
                io_uring_cqe *event = nullptr;
                int status;
                while ((status = io_uring_peek_cqe(&ring_, &event)) == 0) {
                    auto *key = static_cast<Request *>(io_uring_cqe_get_data(event));
                    auto found = active.find(key);
                    if (found == active.end()) throw std::runtime_error("unknown io_uring completion");
                    Completion result{event->res, event->big_cqe[0]};
                    auto request = found->second;
                    active.erase(found);
                    io_uring_cqe_seen(&ring_, event);
                    request->finish(result);
                    completed = true;
                }
                if (status != -EAGAIN && status != -EINTR)
                    throw std::system_error(-status, std::generic_category(), "io_uring_peek_cqe");
                if (completed) continue;
                pollfd descriptors[2] = {{wake_, POLLIN, 0}, {ring_.ring_fd, POLLIN, 0}};
                int ready;
                do { ready = poll(descriptors, 2, -1); } while (ready < 0 && errno == EINTR);
                if (ready < 0) throw std::system_error(errno, std::generic_category(), "io_uring poll");
                if (descriptors[0].revents & POLLIN) {
                    std::uint64_t signal;
                    while (read(wake_, &signal, sizeof(signal)) < 0 && errno == EINTR) {}
                }
                if (descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL))
                    throw std::runtime_error("io_uring completion fd failed");
            }
            io_uring_queue_exit(&ring_);
        } catch (...) {
            auto failure = std::current_exception();
            {
                std::lock_guard<std::mutex> guard(mutex_);
                failure_ = failure;
                for (auto &request : pending_) request->fail(failure);
                pending_.clear();
            }
            space_.notify_all();
            while (std::any_of(active.begin(), active.end(), [](const auto &entry) {
                return entry.second->submitted;
            })) {
                io_uring_cqe *event = nullptr;
                if (io_uring_wait_cqe(&ring_, &event) < 0) {
                    usleep(1000);
                    continue;
                }
                auto found = active.find(static_cast<Request *>(io_uring_cqe_get_data(event)));
                if (found != active.end()) found->second->submitted = false;
                io_uring_cqe_seen(&ring_, event);
            }
            io_uring_queue_exit(&ring_);
            for (auto &entry : active) entry.second->fail(failure);
        }
    }

    io_uring ring_ = {};
    int wake_ = -1;
    std::mutex mutex_;
    std::condition_variable space_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::exception_ptr failure_;
    bool stopping_ = false;
    std::thread worker_;
};

Dispatcher &dispatcher()
{
    static Dispatcher instance;
    return instance;
}

}

UringNvmeRequest uring_submit_nvme(int descriptor, const nvme_passthru_cmd &command,
                                  std::vector<std::uint8_t> payload)
{
    if ((command.nsid != 3 && command.metadata) || command.metadata_len)
        throw std::invalid_argument("NVMe separate metadata is not supported by this transport");
    if (command.data_len != payload.size()) throw std::invalid_argument("NVMe payload length mismatch");
    auto request = std::make_shared<Request>(Operation::Execute, descriptor, payload.size());
    request->owned_data = std::make_shared<std::vector<std::uint8_t>>(std::move(payload));
    std::memcpy(&request->command, &command, offsetof(nvme_passthru_cmd, result));
    request->data = request->owned_data->empty() ? nullptr : request->owned_data->data();
    request->command.addr = reinterpret_cast<std::uint64_t>(request->data);
    return UringNvmeRequest(dispatcher().submit(request).share(), request->owned_data);
}

int uring_execute(int descriptor, nvme_passthru_cmd &command)
{
    if ((command.nsid != 3 && command.metadata) || command.metadata_len)
        throw std::invalid_argument("NVMe separate metadata is not supported by this transport");
    if (command.data_len && !command.addr) throw std::invalid_argument("missing NVMe payload");
    auto request = std::make_shared<Request>(Operation::Execute, descriptor, command.data_len);
    static_assert(offsetof(nvme_uring_cmd, timeout_ms) == offsetof(nvme_passthru_cmd, timeout_ms));
    static_assert(sizeof(nvme_uring_cmd) == 72);
    std::memcpy(&request->command, &command, offsetof(nvme_passthru_cmd, result));
    request->data = reinterpret_cast<void *>(command.addr);
    request->command.addr = reinterpret_cast<std::uint64_t>(request->data);
    auto completion = dispatcher().execute(request);
    command.result = static_cast<std::uint32_t>(completion.result);
    return completion.status;
}

ssize_t uring_read(int descriptor, void *data, std::size_t size, std::uint64_t offset)
{
    auto request = std::make_shared<Request>(Operation::Read, descriptor, size);
    request->data = data;
    request->offset = offset;
    auto completion = dispatcher().execute(request);
    if (static_cast<std::size_t>(completion.status) > size) throw std::runtime_error("oversized read completion");
    return completion.status;
}

ssize_t uring_write(int descriptor, const void *data, std::size_t size, std::uint64_t offset)
{
    auto request = std::make_shared<Request>(Operation::Write, descriptor, size);
    request->data = const_cast<void *>(data);
    request->offset = offset;
    return dispatcher().execute(request).status;
}

ssize_t uring_copy(int source, int destination, std::size_t size,
                   std::uint64_t source_offset, std::uint64_t destination_offset)
{
    auto request = std::make_shared<Request>(Operation::Copy, destination, size);
    request->source = fcntl(source, F_DUPFD_CLOEXEC, 0);
    if (request->source < 0) throw std::system_error(errno, std::generic_category(), "dup copy source");
    request->offset = destination_offset;
    request->source_offset = source_offset;
    return dispatcher().execute(request).status;
}

#else
UringNvmeRequest uring_submit_nvme(int, const nvme_passthru_cmd &, std::vector<std::uint8_t>)
{
    throw std::runtime_error("io_uring backend was not compiled");
}

int uring_execute(int, nvme_passthru_cmd &)
{
    throw std::runtime_error("io_uring backend was not compiled");
}
ssize_t uring_read(int, void *, std::size_t, std::uint64_t)
{
    throw std::runtime_error("io_uring backend was not compiled");
}
ssize_t uring_write(int, const void *, std::size_t, std::uint64_t)
{
    throw std::runtime_error("io_uring backend was not compiled");
}
ssize_t uring_copy(int, int, std::size_t, std::uint64_t, std::uint64_t)
{
    throw std::runtime_error("io_uring backend was not compiled");
}
#endif

}
