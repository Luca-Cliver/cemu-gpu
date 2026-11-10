cmake_minimum_required(VERSION 3.16)

project(cemu_instattention_guest LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)
include(CheckCXXSourceCompiles)
find_path(CEMU_URING_INCLUDE_DIR liburing.h)
find_library(CEMU_URING_LIBRARY uring)
if(CEMU_URING_INCLUDE_DIR AND CEMU_URING_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${CEMU_URING_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${CEMU_URING_LIBRARY}")
    check_cxx_source_compiles("
        #include <liburing.h>
        #include <linux/nvme_ioctl.h>
        int main() {
            io_uring ring{};
            nvme_uring_cmd command{};
            (void)command;
            (void)NVME_URING_CMD_IO;
            (void)IORING_OP_URING_CMD;
            return io_uring_queue_init(64, &ring,
                IORING_SETUP_SQE128 | IORING_SETUP_CQE32);
        }" CEMU_HAVE_URING)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
endif()

add_library(cemu_io_transport STATIC guest/io_transport.cpp)
target_compile_features(cemu_io_transport PUBLIC cxx_std_17)
target_compile_options(cemu_io_transport PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(cemu_io_transport PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cemu_io_transport PUBLIC Threads::Threads)
if(CEMU_HAVE_URING)
    target_compile_definitions(cemu_io_transport PRIVATE CEMU_HAVE_URING=1)
    target_include_directories(cemu_io_transport SYSTEM PRIVATE "${CEMU_URING_INCLUDE_DIR}")
    target_link_libraries(cemu_io_transport PUBLIC "${CEMU_URING_LIBRARY}")
    message(STATUS "CEMU io_uring transport enabled (opt-in CEMU_IO_BACKEND=uring)")
else()
    message(STATUS "CEMU io_uring transport unavailable; ioctl transport remains available")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE PYBIND11_PYTHON_RESULT
)
if(NOT PYBIND11_PYTHON_RESULT EQUAL 0)
    set(PYBIND11_PYTHON_INCLUDE_DIR "")
endif()
find_path(CEMU_PYBIND11_INCLUDE_DIR pybind11/pybind11.h
    HINTS "${PYBIND11_PYTHON_INCLUDE_DIR}"
)
if(NOT EXISTS "${CEMU_PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR
        "pybind11 headers not found. Install pybind11 for ${Python3_EXECUTABLE}, "
        "or set CEMU_PYBIND11_INCLUDE_DIR to the directory containing pybind11/pybind11.h.")
endif()
message(STATUS "Using pybind11 headers: ${CEMU_PYBIND11_INCLUDE_DIR}")

Python3_add_library(_cemu_client MODULE
    guest/cemu_client.cpp
    guest/pybind_module.cpp
    ../util.c
)
target_link_libraries(_cemu_client PRIVATE cemu_io_transport)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_io_transport guest/test_io_transport.cpp guest/cemu_client.cpp ../util.c)
    target_link_libraries(test_io_transport PRIVATE cemu_io_transport)
    add_test(NAME io_transport COMMAND test_io_transport)
    set_tests_properties(io_transport PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(test_client_profile guest/test_client_profile.cpp guest/cemu_client.cpp ../util.c)
    target_link_libraries(test_client_profile PRIVATE cemu_io_transport)
    target_link_options(test_client_profile PRIVATE "-Wl,--wrap=ioctl" "-Wl,--wrap=pread")
    add_test(NAME client_profile COMMAND test_client_profile)
endif()

target_include_directories(_cemu_client PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CEMU_PYBIND11_INCLUDE_DIR}"
)

if(Python3_SOABI)
    set_target_properties(_cemu_client PROPERTIES
        SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    )
endif()

target_compile_features(_cemu_client PRIVATE cxx_std_17)
set_target_properties(_cemu_client PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_options(_cemu_client PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-fPIC>
)
