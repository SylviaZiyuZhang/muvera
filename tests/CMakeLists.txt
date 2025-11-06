# Address Sanitizer
# set(ASAN_COMPILE_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer" "-g")
#s et(ASAN_LINK_FLAGS "-fsanitize=address")

add_executable(fde_test
    fde_test.cpp
)
# target_compile_options(fde_test PRIVATE ${ASAN_COMPILE_FLAGS})
# target_link_options(fde_test PRIVATE ${ASAN_LINK_FLAGS})
target_link_libraries(fde_test
    muvera
    Boost::program_options
    ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
    ${DISKANN_ASYNC_LIB}

    # MKL libraries needed for static diskann
    -Wl,--start-group
    /usr/lib/x86_64-linux-gnu/libmkl_intel_ilp64.a
    /usr/lib/x86_64-linux-gnu/libmkl_core.a
    /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.a
    /usr/lib/x86_64-linux-gnu/libmkl_def.so
    -liomp5
    -Wl,--end-group

    pthread
    m
    dl
)
target_include_directories(fde_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME FDETest COMMAND fde_test)

add_executable(fde_test_static
    fde_test.cpp
)
# target_compile_options(fde_test_static PRIVATE ${ASAN_COMPILE_FLAGS})
# target_link_options(fde_test_static PRIVATE ${ASAN_LINK_FLAGS})
target_link_libraries(fde_test_static
    muvera_static

    # MKL libraries needed for static diskann
    -Wl,--start-group
    /usr/lib/x86_64-linux-gnu/libmkl_intel_ilp64.a
    /usr/lib/x86_64-linux-gnu/libmkl_core.a
    /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.a
    /usr/lib/x86_64-linux-gnu/libmkl_def.so
    -liomp5
    -Wl,--end-group

    pthread
    m
    dl
)
target_include_directories(fde_test_static
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME FDETestStaticLib COMMAND fde_test_static)

add_executable(retriever_test
    retriever_test.cpp
)
target_compile_options(retriever_test PRIVATE ${ASAN_COMPILE_FLAGS})
target_link_options(retriever_test PRIVATE ${ASAN_LINK_FLAGS})
target_link_libraries(retriever_test
    muvera
    Boost::program_options
    ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
    ${DISKANN_ASYNC_LIB}

    -Wl,--start-group
    /usr/lib/x86_64-linux-gnu/libmkl_intel_ilp64.a
    /usr/lib/x86_64-linux-gnu/libmkl_core.a
    /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.a
    /usr/lib/x86_64-linux-gnu/libmkl_def.so
    -liomp5
    -Wl,--end-group

    pthread
    m
    dl
)
target_include_directories(retriever_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME RetrieverTest COMMAND retriever_test)

add_executable(retriever_test_static
    retriever_test.cpp
)
target_compile_options(retriever_test_static PRIVATE ${ASAN_COMPILE_FLAGS})
target_link_options(retriever_test_static PRIVATE ${ASAN_LINK_FLAGS})
target_link_libraries(retriever_test_static
    muvera_static

    # MKL libraries needed for static diskann
    -Wl,--start-group
    /usr/lib/x86_64-linux-gnu/libmkl_intel_ilp64.a
    /usr/lib/x86_64-linux-gnu/libmkl_core.a
    /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.a
    /usr/lib/x86_64-linux-gnu/libmkl_def.so
    -liomp5
    -Wl,--end-group

    pthread
    m
    dl
)
target_include_directories(retriever_test_static
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME RetrieverTestStaticLib COMMAND retriever_test_static)