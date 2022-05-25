cmake_minimum_required(VERSION 3.20)

project(cusparse_csr VERSION 1.0 LANGUAGES CUDA CXX C)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

set_source_files_properties("utils/*" PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} main.cu.cc utils/matrix/dense.cu.cc utils/matrix/sparse.cu.cc utils/spmm/spmm.cu.cc)

target_include_directories(${PROJECT_NAME}
        PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
        PUBLIC ${PROJECT_SOURCE_DIR}/utils)

target_link_libraries(${PROJECT_NAME}
        PUBLIC CUDA::cudart CUDA::cusparse)
