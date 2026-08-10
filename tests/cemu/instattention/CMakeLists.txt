cmake_minimum_required(VERSION 3.16)

project(cemu_instattention_guest LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_CMAKE_RESULT
    )
    if(PYBIND11_CMAKE_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(_cemu_client
    guest/cemu_client.cpp
    guest/pybind_module.cpp
    ../util.c
)

target_include_directories(_cemu_client PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

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
