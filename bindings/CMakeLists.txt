pybind11_add_module(_your_project your_project_bindings.cpp)
target_link_libraries(_your_project PRIVATE your_project)
target_include_directories(_your_project PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()

add_executable(fde_test test/fde_test.cpp)
target_include_directories(fde_test PRIVATE src)
add_test(NAME FDETest COMMAND fde_test)