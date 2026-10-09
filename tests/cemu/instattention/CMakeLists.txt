cmake_minimum_required(VERSION 3.16)

project(cemu_instattention_guest LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

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
