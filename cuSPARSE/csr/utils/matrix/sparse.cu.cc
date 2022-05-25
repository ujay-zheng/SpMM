#include "matrix/sparse.h"

SparseMatrix::SparseMatrix(const int &rows, const int &columns, const int &br, const int &bc, const int &non_zeros,
                           int *row_offsets, int *column_indices, float *values):
        SparseMatrix(rows, columns, non_zeros, br, bc){
    row_offsets_ = new int[(rows_+1)];
    column_indices_ = new int[non_zeros_];
    values_ = new float[non_zeros_];

    memcpy(row_offsets_, row_offsets, sizeof(int)*(rows_+1));
    memcpy(column_indices_, column_indices, sizeof(int)*non_zeros_);
    memcpy(values_, values, sizeof(float)*non_zeros_);
}

SparseMatrix::SparseMatrix(const CudaSparseMatrix& cuda_sparse):
        SparseMatrix(cuda_sparse.Rows(), cuda_sparse.Columns(), cuda_sparse.NonZeros(), cuda_sparse.Br(), cuda_sparse.Bc()){
    row_offsets_ = new int[(rows_+1)];
    column_indices_ = new int[non_zeros_];
    values_ = new float[non_zeros_];

    CHECK_CUDA(cudaMemcpy(row_offsets_, cuda_sparse.RowOffsets(), sizeof(int)*(rows_+1), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(column_indices_, cuda_sparse.ColumnIndices(), sizeof(int)*non_zeros_, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(values_, cuda_sparse.Values(), sizeof(float)*non_zeros_, cudaMemcpyDeviceToHost))
}


CudaSparseMatrix::CudaSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                           const int &non_zeros, int *row_offsets, int *column_indices, float *values):
        CudaSparseMatrix(rows, columns, non_zeros, br, bc ){
    CHECK_CUDA(cudaMalloc((void **)&row_offsets_, sizeof(int)*(rows_+1)))
    CHECK_CUDA(cudaMalloc((void **)column_indices_, sizeof(int)*non_zeros_))
    CHECK_CUDA(cudaMalloc((void **)values_, sizeof(float)*non_zeros_))

    CHECK_CUDA(cudaMemcpy(row_offsets_, row_offsets, sizeof(int)*(rows_+1), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(column_indices_, column_indices, sizeof(int)*non_zeros_, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(values_, values, sizeof(float)*non_zeros_, cudaMemcpyHostToDevice))

    CHECK_CUSPARSE( cusparseCreateCsr(&sparse_matrix_description_, rows_, columns_, non_zeros_,
                                      row_offsets_, column_indices_, values_,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
}

CudaSparseMatrix::CudaSparseMatrix(const SparseMatrix &host_sparse):
        CudaSparseMatrix(host_sparse.Rows(), host_sparse.Columns(), host_sparse.NonZeros(),
                         host_sparse.Br(), host_sparse.Bc()){
    CHECK_CUDA(cudaMalloc((void **)&row_offsets_, sizeof(int)*(rows_+1)))
    CHECK_CUDA(cudaMalloc((void **)column_indices_, sizeof(int)*non_zeros_))
    CHECK_CUDA(cudaMalloc((void **)values_, sizeof(float)*non_zeros_))

    CHECK_CUDA(cudaMemcpy(row_offsets_, host_sparse.RowOffsets(), sizeof(int)*(rows_+1), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(column_indices_, host_sparse.ColumnIndices(), sizeof(int)*non_zeros_, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(values_, host_sparse.Values(), sizeof(float)*non_zeros_, cudaMemcpyHostToDevice))

    CHECK_CUSPARSE( cusparseCreateCsr(&sparse_matrix_description_, rows_, columns_, non_zeros_,
                                      row_offsets_, column_indices_, values_,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
}

CudaSparseMatrix::CudaSparseMatrix(const CudaMatrix &cuda_dense, int br, int bc){
    rows_ = cuda_dense.Rows();
    columns_ = cuda_dense.Columns();
    br_ = br;
    bc_ = bc;
    total_num_ = rows_*columns_;

    cusparseHandle_t handle = nullptr;

    void* dBuffer = nullptr;
    size_t bufferSize = 0;

    CHECK_CUDA( cudaMalloc((void**) &row_offsets_, (rows_ + 1) * sizeof(int)))

    CHECK_CUSPARSE( cusparseCreate(&handle))
    CHECK_CUSPARSE( cusparseCreateCsr(&sparse_matrix_description_, rows_, columns_, 0,
                                      row_offsets_, nullptr, nullptr,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
            handle, cuda_dense.DenseMatrixDescription(), sparse_matrix_description_,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, cuda_dense.DenseMatrixDescription(), sparse_matrix_description_,
                                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer) )

    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(sparse_matrix_description_, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    non_zeros_ = static_cast<int>(nnz);
    sparsity_ = static_cast<float>(total_num_-non_zeros_)/static_cast<float>(total_num_);
    CHECK_CUDA( cudaMalloc((void**) &column_indices_, non_zeros_ * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &values_,  non_zeros_ * sizeof(float)))

    CHECK_CUSPARSE( cusparseCsrSetPointers(sparse_matrix_description_, row_offsets_,
                                           column_indices_, values_))
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, cuda_dense.DenseMatrixDescription(), sparse_matrix_description_,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))

    CHECK_CUSPARSE( cusparseDestroy(handle))
    CHECK_CUDA( cudaFree(dBuffer))

}
