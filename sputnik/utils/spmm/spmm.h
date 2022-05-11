#ifndef SPUTNIK_SPMM_H
#define SPUTNIK_SPMM_H

#include <string>
#include "matrix/matrix_utils.h"

/**
 * @brief By reading the metadata of the mtx file,
 * we can new two pointer, one is point to CudaSparseMatrix Object, the other point to CudaDenseMatrix.
 * @param mtx path to the mtx file
 * @param cuda_sparse_matrix a pointer to CudaSparseMatrix which must be nullptr
 * @param cuda_dense_matrix a pointer to CudaDenseMatrix which must be nullptr
 * */
void cuda_construct_from_mtx(const std::string &mtx, CudaSparseMatrix **cuda_sparse_matrix,
                             CudaDenseMatrix **cuda_dense_matrix);
/**
 * @brief Perform spmm and record GPU runtime
 * @param cuda_sparse_matrix a pointer to CudaSparseMatrix which can not be nullptr
 * @param cuda_dense_matrix a pointer to CudaDenseMatrix which can not be nullptr
 * */
float spmm_timer(CudaSparseMatrix *cuda_sparse_matrix, CudaDenseMatrix *cuda_dense_matrix);
/**
 * @brief Latency measurement for a series of matrix data
 * @param guide path to a guide file which record the mtx path
 * @param out_csv path to csv which will save the result
 * */
void pipeline(const std::string &guide, const std::string &out_csv);

#endif //SPUTNIK_SPMM_H
