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
endif()

if(NOT MSVC)
    find_package(Boost COMPONENTS program_options)
endif()

# For Windows, fall back to nuget version if find_package didn't find it.
if (MSVC AND NOT Boost_FOUND)
    set(DISKANN_BOOST_INCLUDE "${DISKANN_MSVC_PACKAGES}/boost/lib/native/include")
    # Multi-threaded static library.
    set(PROGRAM_OPTIONS_LIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_program_options-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_program_options-vc${MSVC_TOOLSET_VERSION}-mt-x64-*.lib")
    file(GLOB DISKANN_BOOST_PROGRAM_OPTIONS_LIB ${PROGRAM_OPTIONS_LIB_PATTERN})

    set(PROGRAM_OPTIONS_DLIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_program_options-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_program_options-vc${MSVC_TOOLSET_VERSION}-mt-gd-x64-*.lib")
    file(GLOB DISKANN_BOOST_PROGRAM_OPTIONS_DLIB ${PROGRAM_OPTIONS_DLIB_PATTERN})

    if (EXISTS ${DISKANN_BOOST_INCLUDE} AND EXISTS ${DISKANN_BOOST_PROGRAM_OPTIONS_LIB} AND EXISTS ${DISKANN_BOOST_PROGRAM_OPTIONS_DLIB})
        set(Boost_FOUND ON)
        set(Boost_INCLUDE_DIR ${DISKANN_BOOST_INCLUDE})
        add_library(Boost::program_options STATIC IMPORTED)
        set_target_properties(Boost::program_options PROPERTIES IMPORTED_LOCATION_RELEASE "${DISKANN_BOOST_PROGRAM_OPTIONS_LIB}")
        set_target_properties(Boost::program_options PROPERTIES IMPORTED_LOCATION_DEBUG "${DISKANN_BOOST_PROGRAM_OPTIONS_DLIB}")
        message(STATUS "Falling back to using Boost from the nuget package")
    else()
        message(WARNING "Couldn't find Boost. Was looking for ${DISKANN_BOOST_INCLUDE} and ${PROGRAM_OPTIONS_LIB_PATTERN}")
    endif()
endif()

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()

include_directories(${Boost_INCLUDE_DIR})

#MKL Config
if (MSVC)
    # Only the DiskANN DLL and one of the tools need MKL libraries. Additionally, only a small part of MKL is used.
    # Given that and given that MKL DLLs are huge, use static linking to end up with no MKL DLL dependencies and with
    # significantly smaller disk footprint.
    #
    # The compile options are not modified as there's already an unconditional -DMKL_ILP64 define below
    # for all architectures, which is all that's needed.
    set(DISKANN_MKL_INCLUDE_DIRECTORIES "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/include")
    set(DISKANN_MKL_LIB_PATH "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/win-x64")

    set(DISKANN_MKL_LINK_LIBRARIES
        "${DISKANN_MKL_LIB_PATH}/mkl_intel_ilp64.lib"
        "${DISKANN_MKL_LIB_PATH}/mkl_core.lib"
        "${DISKANN_MKL_LIB_PATH}/mkl_intel_thread.lib")
else()
    # expected path for manual intel mkl installs
    set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so;/opt/intel/oneapi/compiler/latest/lib/libiomp5.so")
    foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
        if (EXISTS ${POSSIBLE_OMP_PATH})
            get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
        endif()
    endforeach()

    if(NOT OMP_PATH)
        message(FATAL_ERROR "Could not find Intel OMP in standard locations; use -DOMP_PATH to specify the install location for your environment")
    endif()
    link_directories(${OMP_PATH})

    set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so;/opt/intel/oneapi/mkl/latest/lib/libmkl_core.so")
    foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
        if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
            get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
        endif()
    endforeach()

    set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
    foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
        if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
            set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
        endif()
    endforeach()
    message(STATUS "MKL_PATH: ${MKL_PATH}")
    if(NOT MKL_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location for your environment")
    elseif(NOT MKL_INCLUDE_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_INCLUDE_PATH to specify the install location for headers for your environment")
    endif()
    if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
    elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
    else()
        message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
    endif()
    link_directories(${MKL_PATH})
    include_directories(${MKL_INCLUDE_PATH})

    # compile flags and link libraries
    add_compile_options(-m64 -Wl,--no-as-needed)
    if (NOT PYBIND)
        link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
    else()
        # static linking for python so as to minimize customer dependency issues
        link_libraries(
                ${MKL_PATH}/libmkl_intel_ilp64.a
                ${MKL_PATH}/libmkl_intel_thread.a
                ${MKL_PATH}/libmkl_core.a
                ${MKL_DEF_SO}
                iomp5
                pthread
                m
                dl
        )
    endif()
endif()

add_definitions(-DMKL_ILP64)

# Section for tcmalloc. The DiskANN tools are always linked to tcmalloc. For Windows, they also need to
# force-include the _tcmalloc symbol for enabling tcmalloc.
#
# The DLL itself needs to be linked to tcmalloc only if DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS
# is enabled.
if (MSVC)
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/gperftools/gperftools.sln")
        message(FATAL_ERROR "The gperftools submodule was not found. "
                "Please check-out git submodules by doing 'git submodule init' followed by 'git submodule update'")
    endif()

    set(TCMALLOC_LINK_LIBRARY "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib")
    set(TCMALLOC_WINDOWS_RUNTIME_FILES
        "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.dll"
        "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.pdb")

    # Tell CMake how to build the tcmalloc linker library from the submodule.
    add_custom_target(build_libtcmalloc_minimal DEPENDS ${TCMALLOC_LINK_LIBRARY})
    add_custom_command(OUTPUT ${TCMALLOC_LINK_LIBRARY}
                        COMMAND ${CMAKE_VS_MSBUILD_COMMAND} gperftools.sln /m /nologo
                            /t:libtcmalloc_minimal /p:Configuration="Release-Patch"
                            /property:Platform="x64"
                            /p:PlatformToolset=v${MSVC_TOOLSET_VERSION}
                            /p:WindowsTargetPlatformVersion=${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/gperftools)
 
    add_library(libtcmalloc_minimal_for_exe STATIC IMPORTED)
    add_library(libtcmalloc_minimal_for_dll STATIC IMPORTED)

    set_target_properties(libtcmalloc_minimal_for_dll PROPERTIES
                          IMPORTED_LOCATION "${TCMALLOC_LINK_LIBRARY}")

    set_target_properties(libtcmalloc_minimal_for_exe PROPERTIES
                          IMPORTED_LOCATION "${TCMALLOC_LINK_LIBRARY}"
                          INTERFACE_LINK_OPTIONS /INCLUDE:_tcmalloc)

    # Ensure libtcmalloc_minimal is built before it's being used.
    add_dependencies(libtcmalloc_minimal_for_dll build_libtcmalloc_minimal)
    add_dependencies(libtcmalloc_minimal_for_exe build_libtcmalloc_minimal)

    set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS libtcmalloc_minimal_for_exe)
elseif(NOT PYBIND)
    set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS "-ltcmalloc")
endif()

if (DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
    add_definitions(-DRELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)

    if (MSVC)
        set(DISKANN_DLL_TCMALLOC_LINK_OPTIONS libtcmalloc_minimal_for_dll)
    endif()
endif()

if (NOT MSVC)
    set(DISKANN_ASYNC_LIB aio)
endif()

#Main compiler/linker settings 
if(MSVC)
	#language options
	add_compile_options(/permissive- /openmp:experimental /Zc:twoPhase- /Zc:inline /WX- /std:c++17 /Gd /W3 /MP /Zi /FC /nologo)
	#code generation options
	add_compile_options(/arch:AVX2 /fp:fast /fp:except- /EHsc /GS- /Gy)
	#optimization options
	add_compile_options(/Ot /Oy /Oi)
	#path options
	add_definitions(-DUSE_AVX2 -DUSE_ACCELERATED_PQ -D_WINDOWS -DNOMINMAX -DUNICODE)
    # Linker options. Exclude VCOMP/VCOMPD.LIB which contain VisualStudio's version of OpenMP.
    # MKL was linked against Intel's OpenMP and depends on the corresponding DLL.
    add_link_options(/NODEFAULTLIB:VCOMP.LIB /NODEFAULTLIB:VCOMPD.LIB /DEBUG:FULL /OPT:REF /OPT:ICF)
	
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
else()
    set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
    if (NOT PYBIND)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -Ofast")
        if (NOT PORTABLE)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        endif()
    else()
        # -Ofast is not supported in a python extension module
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -fPIC")
    endif()
endif()


# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

# External dependencies
add_subdirectory(external/diskann)
file(GLOB DISKANN_SRC
    "${PROJECT_SOURCE_DIR}/external/diskann/src/*.cpp"
)

add_library(muvera SHARED
    src/fde.cpp
    src/exact_chamfer_retriever.cpp
    ${DISKANN_SRC}
)

target_link_libraries(muvera
    diskann
    Boost::program_options
    ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
    ${DISKANN_ASYNC_LIB}
)

target_include_directories(muvera
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

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    add_subdirectory(bindings)
endif()

# Tests (optional)
enable_testing()
add_subdirectory(tests)
