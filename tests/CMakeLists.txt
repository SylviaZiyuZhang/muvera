add_executable(fde_test
    fde_test.cpp
    ${DISKANN_SRC}
)
target_link_libraries(fde_test
    muvera
    diskann
    Boost::program_options
    ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
    ${DISKANN_ASYNC_LIB}
)
target_include_directories(fde_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME FDETest COMMAND fde_test)

add_executable(retriever_test
    retriever_test.cpp
    ${DISKANN_SRC}
)
target_link_libraries(retriever_test
    muvera
    diskann
    Boost::program_options
    ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
    ${DISKANN_ASYNC_LIB}
)
target_include_directories(retriever_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external/diskann/include
)
add_test(NAME RetrieverTest COMMAND retriever_test)