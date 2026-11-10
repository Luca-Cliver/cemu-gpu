#include "cemu_client.h"
#include "io_transport.h"
#include "../../util.h"

#include <cstring>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using cemu::instattention::CemuClient;
using cemu::instattention::ExecuteOptions;
using cemu::instattention::IoProfile;
using cemu::instattention::MemoryRange;
using cemu::instattention::ProgramTarget;

namespace {

std::uint64_t profile_clock_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

py::dict profile_result(const IoProfile &timing, std::uint64_t start,
                        std::uint64_t prepared)
{
    const auto finished = profile_clock_ns();
    py::dict result;
    result["binding_prepare_wall"] = prepared - start;
    result["buffer_prepare_wall"] = timing.prepare_ns;
    result["io_wait_wall"] = timing.wait_ns;
    result["buffer_finish_wall"] = timing.finish_ns;
    result["binding_finish_wall"] = finished - prepared - timing.prepare_ns - timing.wait_ns - timing.finish_ns;
    result["io_bytes"] = timing.bytes;
    return result;
}

std::size_t contiguous_buffer_size(const py::buffer_info &info)
{
    if (info.itemsize <= 0 || info.ndim < 0) {
        throw py::value_error("invalid Python buffer");
    }

    std::size_t element_count = 1;
    py::ssize_t expected_stride = info.itemsize;
    for (py::ssize_t dimension = info.ndim; dimension-- > 0;) {
        if (info.shape[dimension] < 0 || info.strides[dimension] < 0) {
            throw py::value_error("negative shapes and strides are not supported");
        }
        if (info.shape[dimension] > 1 && info.strides[dimension] != expected_stride) {
            throw py::value_error("the Python buffer must be C-contiguous");
        }
        if (info.shape[dimension] == 0) {
            return 0;
        }
        const auto shape = static_cast<std::size_t>(info.shape[dimension]);
        if (element_count > std::numeric_limits<std::size_t>::max() / shape) {
            throw std::overflow_error("Python buffer size overflow");
        }
        element_count *= shape;
        if (expected_stride > std::numeric_limits<py::ssize_t>::max() /
                                  info.shape[dimension]) {
            throw std::overflow_error("Python buffer stride overflow");
        }
        expected_stride *= info.shape[dimension];
    }

    const auto item_size = static_cast<std::size_t>(info.itemsize);
    if (element_count > std::numeric_limits<std::size_t>::max() / item_size) {
        throw std::overflow_error("Python buffer byte size overflow");
    }
    return element_count * item_size;
}

std::vector<std::uint8_t> copy_python_buffer(const py::buffer &buffer)
{
    const py::buffer_info info = buffer.request();
    const std::size_t size = contiguous_buffer_size(info);
    std::vector<std::uint8_t> result(size);
    if (size != 0) {
        std::memcpy(result.data(), info.ptr, size);
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_cemu_client, module)
{
    module.doc() = "Guest-side Python bindings for CEMU computational-storage commands";
    module.attr("IO_PROFILE_VERSION") = 1;
    module.def("guest_clock_snapshot", []() {
        const auto steady = std::chrono::steady_clock::now();
        const auto high_resolution = std::chrono::high_resolution_clock::now();
        const auto system = std::chrono::system_clock::now();
        return py::make_tuple(
            std::chrono::duration_cast<std::chrono::nanoseconds>(steady.time_since_epoch()).count(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(high_resolution.time_since_epoch()).count(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(system.time_since_epoch()).count());
    });
    module.def("guest_clock_info", []() {
        py::dict result;
        result["steady_is_steady"] = std::chrono::steady_clock::is_steady;
        result["high_resolution_is_steady"] = std::chrono::high_resolution_clock::is_steady;
        result["high_resolution_is_system"] = std::is_same_v<
            std::chrono::high_resolution_clock, std::chrono::system_clock>;
        result["high_resolution_is_steady_clock"] = std::is_same_v<
            std::chrono::high_resolution_clock, std::chrono::steady_clock>;
        return result;
    });
    module.def("io_uring_available", &cemu::instattention::io_uring_available);
    using cemu::instattention::UringNvmeRequest;
    py::class_<UringNvmeRequest>(module, "UringNvmeRequest")
        .def("done", &UringNvmeRequest::done)
        .def("result", [](const UringNvmeRequest &request) {
            cemu::instattention::NvmeCompletion completion;
            {
                py::gil_scoped_release release;
                completion = request.result();
            }
            const auto &data = request.data();
            return py::make_tuple(completion.status, completion.result,
                py::bytes(reinterpret_cast<const char *>(data.data()), data.size()));
        });
    module.def("uring_wait_any", &cemu::instattention::uring_wait_any,
               py::call_guard<py::gil_scoped_release>());
    module.def("uring_nvme_submit", [](int descriptor, py::bytes packed_command,
                                        py::bytes payload) {
        std::string raw = packed_command;
        std::string data = payload;
        if (raw.size() != sizeof(nvme_passthru_cmd))
            throw py::value_error("invalid NVMe passthrough structure size");
        nvme_passthru_cmd command = {};
        std::memcpy(&command, raw.data(), sizeof(command));
        std::vector<std::uint8_t> owned(data.begin(), data.end());
        py::gil_scoped_release release;
        return cemu::instattention::uring_submit_nvme(descriptor, command, std::move(owned));
    });
    module.def("uring_nvme", [](int descriptor, py::bytes packed_command,
                                 py::bytes payload) {
        std::string raw = packed_command;
        std::string data = payload;
        if (raw.size() != sizeof(nvme_passthru_cmd))
            throw py::value_error("invalid NVMe passthrough structure size");
        nvme_passthru_cmd command = {};
        std::memcpy(&command, raw.data(), sizeof(command));
        if (command.data_len != data.size())
            throw py::value_error("NVMe payload length mismatch");
        command.addr = data.empty() ? 0 : reinterpret_cast<std::uint64_t>(data.data());
        int status;
        {
            py::gil_scoped_release release;
            status = cemu::instattention::uring_execute(descriptor, command);
        }
        return py::make_tuple(status, command.result, py::bytes(data));
    }, py::arg("descriptor"), py::arg("command"), py::arg("payload"));
    module.def("uring_copy_file_range", &cemu::instattention::uring_copy,
               py::arg("source_fd"), py::arg("destination_fd"), py::arg("size"),
               py::arg("offset_src"), py::arg("offset_dst"),
               py::call_guard<py::gil_scoped_release>());

    py::enum_<ProgramTarget>(module, "ProgramTarget")
        .value("HOST", ProgramTarget::Host)
        .value("CUDA_DEVICE_POINTER", ProgramTarget::CudaDevicePointer)
        .export_values();

    module.def(
        "get_file_extents",
        [](const std::string &path, std::uint64_t size) {
            if (size == 0 || size % LBA_SIZE != 0) {
                throw py::value_error("file extent size must be a positive multiple of 512");
            }
            if (size / LBA_SIZE > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw py::value_error("file extent request is too large");
            }
            std::uint64_t mapped_size = size;
            NvmeCopyFormat2 *descriptors = nullptr;
            int blocks = static_cast<int>(size / LBA_SIZE);
            int descriptor_count = 0;
            int result = get_file_mappings(path.c_str(), &mapped_size, &descriptors,
                                           1, &blocks, &descriptor_count);
            std::unique_ptr<NvmeCopyFormat2, decltype(&std::free)>
                owned(descriptors, &std::free);
            if (result < 0 || result != descriptor_count || mapped_size != size) {
                throw std::runtime_error("cannot resolve complete NVM file extents: " + path);
            }
            py::list extents;
            for (int index = 0; index < descriptor_count; ++index) {
                extents.append(py::make_tuple(
                    descriptors[index].slba,
                    static_cast<std::uint32_t>(descriptors[index].nlb) + 1));
            }
            return extents;
        },
        py::arg("path"), py::arg("size"));

    py::class_<MemoryRange>(module, "MemoryRange")
        .def(py::init<std::string, std::uint64_t, std::uint64_t>(),
             py::arg("path"), py::arg("offset"), py::arg("size"))
        .def_readwrite("path", &MemoryRange::path)
        .def_readwrite("offset", &MemoryRange::offset)
        .def_readwrite("size", &MemoryRange::size);

    py::class_<CemuClient>(module, "CemuClient")
        .def(py::init<std::string, std::string>(),
             py::arg("control_path") = "/dev/nvme0c3",
             py::arg("namespace_path") = "/dev/ng0n3")
        .def("load_program", &CemuClient::load_program,
             py::arg("program_name"),
             py::arg("shared_library_path"),
             py::arg("function_name"),
             py::arg("target") = ProgramTarget::Host,
             py::arg("runtime") = 0,
             py::arg("runtime_scale_tenths") = 0,
             py::arg("indirect") = false,
             py::arg("replace_existing") = false)
        .def("activate_program", &CemuClient::activate_program)
        .def("create_memory_ranges", &CemuClient::create_memory_ranges,
             py::arg("ranges"))
        .def("release_memory_ranges", &CemuClient::release_memory_ranges,
             py::call_guard<py::gil_scoped_release>())
        .def("bind_fdm_kv_buffer",
             [](CemuClient &client, const py::buffer &metadata) {
                 const py::buffer_info info = metadata.request();
                 const std::size_t size = contiguous_buffer_size(info);
                 py::gil_scoped_release release;
                 client.bind_fdm_kv_buffer(info.ptr, size);
             }, py::arg("metadata"))
        .def("write_range",
             [](CemuClient &client, std::size_t range_index,
                const py::buffer &buffer, std::uint64_t range_offset, bool profile) -> py::object {
                 const auto started = profile ? profile_clock_ns() : 0;
                 const py::buffer_info info = buffer.request();
                 const std::size_t size = contiguous_buffer_size(info);
                 const auto prepared = profile ? profile_clock_ns() : 0;
                 IoProfile timing;
                 {
                     py::gil_scoped_release release;
                     client.write_range(range_index, info.ptr, size, range_offset, profile ? &timing : nullptr);
                 }
                 if (profile) return py::make_tuple(py::none(), profile_result(timing, started, prepared));
                 return py::none();
             },
             py::arg("range_index"), py::arg("data"),
             py::arg("range_offset") = 0, py::kw_only(), py::arg("profile") = false)
        .def("read_range",
             [](const CemuClient &client, std::size_t range_index,
                std::size_t size, std::uint64_t range_offset, bool profile) -> py::object {
                 const auto started = profile ? profile_clock_ns() : 0;
                 std::vector<std::uint8_t> data;
                 IoProfile timing;
                 const auto prepared = profile ? profile_clock_ns() : 0;
                 {
                     py::gil_scoped_release release;
                     data = client.read_range(range_index, size, range_offset, profile ? &timing : nullptr);
                 }
                 py::array_t<std::uint8_t> result(data.size());
                 if (!data.empty()) {
                     std::memcpy(result.mutable_data(), data.data(), data.size());
                 }
                 if (profile) return py::make_tuple(result, profile_result(timing, started, prepared));
                 return result;
             },
             py::arg("range_index"), py::arg("size"),
             py::arg("range_offset") = 0, py::kw_only(), py::arg("profile") = false)
        .def("read_range_into",
             [](const CemuClient &client, std::size_t range_index,
                const py::buffer &buffer, std::uint64_t range_offset, bool profile) -> py::object {
                 const auto started = profile ? profile_clock_ns() : 0;
                 const py::buffer_info info = buffer.request(true);
                 if (info.format.find('O') != std::string::npos) {
                     throw py::value_error("object buffers cannot receive raw I/O");
                 }
                 const std::size_t size = contiguous_buffer_size(info);
                 const auto prepared = profile ? profile_clock_ns() : 0;
                 IoProfile timing;
                 {
                     py::gil_scoped_release release;
                     client.read_range_into(range_index, info.ptr, size, range_offset,
                                            profile ? &timing : nullptr);
                 }
                 if (profile) return py::make_tuple(py::none(), profile_result(timing, started, prepared));
                 return py::none();
             }, py::arg("range_index"), py::arg("output"),
             py::arg("range_offset") = 0, py::kw_only(), py::arg("profile") = false)
        .def("read_range_bytes",
             [](const CemuClient &client, std::size_t range_index,
                std::size_t size, std::uint64_t range_offset) {
                 std::vector<std::uint8_t> data;
                 {
                     py::gil_scoped_release release;
                     data = client.read_range(range_index, size, range_offset);
                 }
                 if (data.empty()) {
                     return py::bytes("");
                 }
                 return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
             },
             py::arg("range_index"), py::arg("size"),
             py::arg("range_offset") = 0)
        .def("execute",
             [](CemuClient &client,
                std::uint64_t cparam1,
                std::uint64_t cparam2,
                std::uint32_t group,
                std::uint32_t chunk_nlb,
                std::uint32_t runtime,
                const py::object &metadata, bool profile) -> py::object {
                 const auto started = profile ? profile_clock_ns() : 0;
                 ExecuteOptions options;
                 options.cparam1 = cparam1;
                 options.cparam2 = cparam2;
                 options.group = group;
                 options.chunk_nlb = chunk_nlb;
                 options.runtime = runtime;

                 std::vector<std::uint8_t> metadata_bytes;
                 if (!metadata.is_none()) {
                     metadata_bytes = copy_python_buffer(
                         py::reinterpret_borrow<py::buffer>(metadata));
                 }

                 IoProfile timing;
                 std::uint32_t result;
                 const auto prepared = profile ? profile_clock_ns() : 0;
                 {
                     py::gil_scoped_release release;
                     result = client.execute(
                         options,
                         metadata_bytes.empty() ? nullptr : metadata_bytes.data(),
                         metadata_bytes.size(), profile ? &timing : nullptr);
                 }
                 if (profile) return py::make_tuple(result, profile_result(timing, started, prepared));
                 return py::int_(result);
             },
             py::arg("cparam1") = 0,
             py::arg("cparam2") = 0,
             py::arg("group") = 0,
             py::arg("chunk_nlb") = 0,
             py::arg("runtime") = 0,
             py::arg("metadata") = py::none(), py::kw_only(), py::arg("profile") = false)
        .def("close", &CemuClient::close)
        .def_property_readonly("program_id", &CemuClient::program_id)
        .def_property_readonly("memory_range_set_id",
                               &CemuClient::memory_range_set_id)
        .def_property_readonly("memory_range_count",
                               &CemuClient::memory_range_count)
        .def("__enter__", [](CemuClient &client) -> CemuClient & {
            return client;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](CemuClient &client, const py::object &,
                            const py::object &, const py::object &) {
            client.close();
        });
}
