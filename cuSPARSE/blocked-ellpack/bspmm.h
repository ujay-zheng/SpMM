//
// Created by root on 4/30/22.
//

#ifndef CUSPARSE_BSPMM_H
#define CUSPARSE_BSPMM_H

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CREATE_DENSE_DESCRIBE( dense_describe,d_dense, h_dense, dense_rows, dense_cols,\
                              ld, dense_size, type_size, cuda_mem_order, cuda_value_type)\
{\
    CHECK_CUDA( cudaMalloc((void**)&d_dense, dense_size*type_size))\
    CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * type_size,\
                           cudaMemcpyHostToDevice) )\
    CHECK_CUSPARSE( cusparseCreateDnMat(&dense_describe, dense_rows, dense_cols, ld, d_dense,\
                                        cuda_value_type, cuda_mem_order) )\
}

#define CREATE_SPARSE_DESCRIBE(sparse_describe, h_ell_col, d_ell_col, h_ell_val, d_ell_val,\
                               spare_row, sparse_col, ell_col_num, sparse_val_num,\
                               ell_blocksize, col_type_size, val_type_size,\
                               cuda_ell_idx_type, cuda_value_type)\
{\
    CHECK_CUDA( cudaMalloc((void**) &d_ell_col, sparse_val_num / (ell_blocksize * ell_blocksize) * col_type_size))\
    CHECK_CUDA( cudaMalloc((void**) &d_ell_val, sparse_val_num * val_type_size))\
    CHECK_CUDA( cudaMemcpy(d_ell_col, h_ell_col, \
                            sparse_val_num / (ell_blocksize * ell_blocksize) * col_type_size, cudaMemcpyHostToDevice) )\
    CHECK_CUDA( cudaMemcpy(d_ell_val, h_ell_val, sparse_val_num * val_type_size, cudaMemcpyHostToDevice) )\
    CHECK_CUSPARSE( cusparseCreateBlockedEll(&sparse_describe, spare_row, sparse_col, ell_blocksize, ell_col_num,\
                                             d_ell_col, d_ell_val, cuda_ell_idx_type,  CUSPARSE_INDEX_BASE_ZERO,\
                                             cuda_value_type) )\
}

#define DENSE2BLOCKSPARSE(handle, dense_mat, sparse_mat, d_dense, h_dense, h_columns, d_columns, h_values, d_values,\
                          buffer_temp, rows, cols, ld, dense_size, h_val_type_size, h_col_type_size, ell_col_num,\
                          sparse_val_num, ell_blocksize, cuda_mem_order, cuda_value_type, cuda_ell_idx_type)\
{\
    size_t dense2sparse_bufferSize = 0;\
    CREATE_DENSE_DESCRIBE( dense_mat, d_dense, h_dense, rows, cols,\
                            ld, dense_size, h_val_type_size, cuda_mem_order, cuda_value_type)\
    CREATE_SPARSE_DESCRIBE(sparse_mat, h_columns, d_columns, h_values, d_values, rows, cols, ell_col_num, sparse_val_num,\
                           ell_blocksize, h_col_type_size, h_val_type_size, cuda_ell_idx_type, cuda_value_type)\
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize( handle, dense_mat, sparse_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &dense2sparse_bufferSize) )\
    CHECK_CUDA( cudaMalloc(&buffer_temp, dense2sparse_bufferSize) )\
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, dense_mat, sparse_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_temp) )\
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, dense_mat, sparse_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_temp) )\
}

#define SPMM(handle, opA, opB, alpha, matA, matB, beta, matC, cuda_value_type, alg, spmm_buffer, elapsedTime)\
{\
    size_t spmm_buffer_size = 0;\
    CHECK_CUSPARSE( cusparseSpMM_bufferSize( handle, opA, opB, &alpha, matA, matB, &beta, matC, cuda_value_type, alg, &spmm_buffer_size) )\
    CHECK_CUDA( cudaMalloc(&spmm_buffer, spmm_buffer_size) )\
    cudaEvent_t start, stop;\
    cudaEventCreate(&start);\
    cudaEventCreate(&stop);\
    cudaEventRecord(start, 0);\
    cusparseStatus_t s = cusparseSpMM(handle, opA, opB, &alpha, matA, matB, &beta, matC, cuda_value_type, alg, spmm_buffer);\
    cudaEventRecord(stop, 0);\
    cudaEventSynchronize(start);\
    cudaEventSynchronize(stop);\
    cudaEventElapsedTime(&elapsedTime, start, stop);\
    cudaEventDestroy(start);\
    cudaEventDestroy(stop);\
    CHECK_CUSPARSE(s)\
}


#define DESTORY_MATRIX_DESCRIBE(matA_dense, matA, matB, matC, handle)\
{\
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA_dense) )\
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )\
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )\
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )\
    CHECK_CUSPARSE( cusparseDestroy(handle) )\
}

#define FREE_CUDA_MEMORY(dense2sparse_buffer, spmm_buffer, d_A_columns, d_A_values, dA, dB, dC)\
{\
    CHECK_CUDA( cudaFree(dense2sparse_buffer) )\
    CHECK_CUDA( cudaFree(spmm_buffer) )\
    CHECK_CUDA( cudaFree(d_A_columns) )\
    CHECK_CUDA( cudaFree(d_A_values) )\
    CHECK_CUDA( cudaFree(dA))\
    CHECK_CUDA( cudaFree(dB) )\
    CHECK_CUDA( cudaFree(dC) )\
}

#endif //CUSPARSE_BSPMM_H
