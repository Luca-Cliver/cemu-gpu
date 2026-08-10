#include "cemu_client.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using cemu::instattention::CemuClient;
using cemu::instattention::ExecuteOptions;
using cemu::instattention::MemoryRange;
using cemu::instattention::ProgramTarget;

namespace {

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

    py::enum_<ProgramTarget>(module, "ProgramTarget")
        .value("HOST", ProgramTarget::Host)
        .value("CUDA_DEVICE_POINTER", ProgramTarget::CudaDevicePointer)
        .export_values();

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
        .def("write_range",
             [](CemuClient &client, std::size_t range_index,
                const py::buffer &buffer, std::uint64_t range_offset) {
                 const py::buffer_info info = buffer.request();
                 const std::size_t size = contiguous_buffer_size(info);
                 py::gil_scoped_release release;
                 client.write_range(range_index, info.ptr, size, range_offset);
             },
             py::arg("range_index"), py::arg("data"),
             py::arg("range_offset") = 0)
        .def("read_range",
             [](const CemuClient &client, std::size_t range_index,
                std::size_t size, std::uint64_t range_offset) {
                 std::vector<std::uint8_t> data;
                 {
                     py::gil_scoped_release release;
                     data = client.read_range(range_index, size, range_offset);
                 }
                 py::array_t<std::uint8_t> result(data.size());
                 if (!data.empty()) {
                     std::memcpy(result.mutable_data(), data.data(), data.size());
                 }
                 return result;
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
                const py::object &metadata) {
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

                 py::gil_scoped_release release;
                 return client.execute(
                     options,
                     metadata_bytes.empty() ? nullptr : metadata_bytes.data(),
                     metadata_bytes.size());
             },
             py::arg("cparam1") = 0,
             py::arg("cparam2") = 0,
             py::arg("group") = 0,
             py::arg("chunk_nlb") = 0,
             py::arg("runtime") = 0,
             py::arg("metadata") = py::none())
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
