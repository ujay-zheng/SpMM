#ifndef CU_SPARSE_CSR_SPMM_H
#define CU_SPARSE_CSR_SPMM_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include "matrix/matrix.h"


float latency_evaluate(CudaSparseMatrix *cuda_sparse, CudaMatrix *cuda_dense, cusparseSpMMAlg_t alg, int repeat=100);
void cuda_construct_from_mtx(const std::string &mtx, CudaSparseMatrix **cuda_sparse,
                             CudaMatrix **cuda_dense, cusparseOrder_t layout);
void pipeline(const std::string &guide, const std::string &out_csv, cusparseSpMMAlg_t alg);

#endif //CU_SPARSE_CSR_SPMM_H
