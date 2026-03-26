cmake_minimum_required(VERSION 3.15)
project(muvera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define where CMake modules are located
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# DiskANN core uses header-only libraries. Only DiskANN tools need program_options which has a linker library,
# but its size is small. Reduce number of dependent DLLs by linking statically.
if (MSVC)
    set(Boost_USE_STATIC_LIBS ON)
else()
    find_package(Boost COMPONENTS program_options REQUIRED)
endif()

# External dependencies
add_subdirectory(external/diskann)  # this builds the diskann target


add_library(muvera SHARED
    src/fde.cpp
    src/exact_chamfer_retriever.cpp
    src/relaxed_chamfer_retriever.cpp
    src/muvera_retriever.cpp
)

add_library(muvera_static STATIC
    src/fde.cpp
    src/exact_chamfer_retriever.cpp
    src/relaxed_chamfer_retriever.cpp
    src/muvera_retriever.cpp
)

if(MSVC)
    target_compile_options(muvera INTERFACE /arch:AVX2)
    target_compile_options(muvera_static INTERFACE /arch:AVX2)
else()
    target_compile_options(muvera INTERFACE -mavx2 -mfma -msse2)
    target_compile_options(muvera_static INTERFACE -mavx2 -mfma -msse2)
endif()

target_link_libraries(muvera
    diskann
    Boost::program_options
)

target_link_libraries(muvera_static
    PUBLIC diskann
    PRIVATE Boost::program_options
)

target_include_directories(muvera
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(muvera_static
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Installation (optional)
install(TARGETS muvera
        EXPORT muveraTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

# Tests (optional)
enable_testing()
add_subdirectory(tests)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
