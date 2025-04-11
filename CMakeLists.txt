cmake_minimum_required(VERSION 3.15)
project(muvera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define where CMake modules are located
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

# External dependencies
add_subdirectory(external/diskann)

# Main library
# TODO: FIXME
add_library(muvera SHARED
    src/muvera/module1.cpp
)

target_include_directories(muvera
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(muvera
    PUBLIC
        diskann
)

# Installation (optional)
install(TARGETS muvera
        EXPORT muveraTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    add_subdirectory(bindings)
endif()

# Tests (optional)
enable_testing()
add_subdirectory(tests)
