#ifndef CU_SPARSE_CSR_CUDA_UTILS_H
#define CU_SPARSE_CSR_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cusparse.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#endif //CU_SPARSE_CSR_CUDA_UTILS_H
