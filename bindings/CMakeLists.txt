cmake_minimum_required(VERSION 3.15)

# Use CONFIG to make CMake find the pybind11Config.cmake shipped with pip
# find_package(pybind11 CONFIG REQUIRED)

# Only find pybind11 if it hasn't already been found
if(NOT TARGET pybind11::pybind11)
    # Point explicitly to the pybind11 cmake dir provided by Python
    find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
endif()

# Disable LTO for pybind11 to prevent double target creation
set(PYBIND11_DISABLE_LTO ON CACHE BOOL "Disable LTO" FORCE)

# Add your Python module
pybind11_add_module(muvera_pybind
    pybind_module.cpp
)

# Link to your static library and dependencies
target_link_libraries(muvera_pybind PRIVATE
    muvera_static
)

target_include_directories(muvera_pybind PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/diskann/include
)

# Ensure position independent code for shared module
set_target_properties(muvera_pybind PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "muvera_pybind"              # module name, must match PYBIND11_MODULE name
    PREFIX ""                                # remove "lib" prefix for Python module
)

# Ensure correct Python extension suffix (.so, .pyd)
pybind11_extension(muvera_pybind)
pybind11_strip(muvera_pybind)
