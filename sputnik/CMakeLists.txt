cmake_minimum_required(VERSION 3.10)

project(Sputnik VERSION 1.0 LANGUAGES CUDA CXX C)

enable_language(CUDA)

configure_file(SputnikConfig.h.in SputnikConfig.h)

cmake_policy(SET CMP0048 NEW)

set(CMAKE_CXX_STANDARD 11)

set_source_files_properties("utils/matrix/*" PROPERTIES LANGUAGE CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_executable(${PROJECT_NAME} main.cu.cc utils/matrix/dense.cu.cc utils/matrix/sparse.cu.cc utils/spmm/spmm.cu.cc)

target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}/utils
        PUBLIC ${PROJECT_BINARY_DIR}
        PUBLIC /usr/local/sputnik/include/
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME}
        PUBLIC ${CUDART_LIBRARY}
        PUBLIC /usr/local/sputnik/lib/libsputnik.so)
