#include <cstring>
#include <numeric>
#include <algorithm>
#include "matrix/dense.h"
#include "matrix/sparse.h"


HostSparseMatrix::HostSparseMatrix(
        const int &rows, const int &columns, const int &non_zeros,
        const int &br, const int &bc, const float &sparsity, const std::vector<int> &row_offsets,
        const std::vector<int> &column_indices, const std::vector<float> &values){
    InitFromCSRFormat(rows, columns, non_zeros, br, bc, sparsity, row_offsets, column_indices, values);
}

HostSparseMatrix::HostSparseMatrix(const CudaSparseMatrix &cuda_sparse_matrix)
    : SparseMatrix(cuda_sparse_matrix.Rows(), cuda_sparse_matrix.Columns(), cuda_sparse_matrix.Non_zeros(),
                   cuda_sparse_matrix.BR(), cuda_sparse_matrix.BC(), cuda_sparse_matrix.Sparsity()){
    values_ = new float[non_zeros_];
    column_indices_ = new int[non_zeros_];
    row_offsets_ = new int[(rows_+1)];
    row_indices_ = new int[rows_];

    CHECK_CUDA(cudaMemcpy(values_, cuda_sparse_matrix.Values(),
                          sizeof(float) * non_zeros_,
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(column_indices_, cuda_sparse_matrix.ColumnIndices(),
                          sizeof(int) * non_zeros_,
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(row_offsets_, cuda_sparse_matrix.RowOffsets(),
                          sizeof(int) * (rows_+1),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(row_indices_, cuda_sparse_matrix.RowIndices(),
                          sizeof(int) * rows_,
                          cudaMemcpyDeviceToHost))
}

HostSparseMatrix::HostSparseMatrix(const DenseMatrix &dense_matrix) {
    InitFromDenseMatrix(dense_matrix);
}

HostSparseMatrix::HostSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                                   const float &sparsity, const std::vector<std::vector<float>> &values) {
    DenseMatrix dense_matrix(rows, columns, br, bc, sparsity, values);
    InitFromDenseMatrix(dense_matrix);
}

void HostSparseMatrix::InitFromCSRFormat(
        const int &rows, const int &columns, const int &non_zeros,
        const int &br, const int &bc, const float &sparsity, const std::vector<int> &row_offsets,
        const std::vector<int> &column_indices, const std::vector<float> &values) {
    rows_ = rows;
    columns_ = columns;
    non_zeros_ = non_zeros;
    br_ = br;
    bc_ = bc;
    sparsity_ = sparsity;

    values_ = new float[non_zeros_];
    column_indices_ = new int[non_zeros_];
    row_offsets_ = new int[(rows_+1)];
    row_indices_ = new int[rows_];

    std::memcpy(values_, values.data(),
                non_zeros_ * sizeof(float));
    std::memcpy(column_indices_, column_indices.data(),
                non_zeros * sizeof(int));
    std::memcpy(row_offsets_, row_offsets.data(),
                (rows+1) * sizeof(int));
    DecreasedSortedRowSwizzle();
}

void HostSparseMatrix::DecreasedSortedRowSwizzle(){
    int *row_offsets=row_offsets_;
    std::vector<int> swizzle_staging(rows_);
    std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

    std::sort(swizzle_staging.begin(), swizzle_staging.end(),
              [&row_offsets](int idx_a, int idx_b) {
                  int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
                  int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
                  return length_a > length_b;
              });

    std::memcpy(row_indices_, swizzle_staging.data(), sizeof(int) * rows_);
}

void HostSparseMatrix::InitFromDenseMatrix(const DenseMatrix &dense_matrix) {
    int non_zeros=0;
    std::vector<int> row_offsets(dense_matrix.Rows()+1);
    row_offsets[0] = 0;
    for(int i=0;i<dense_matrix.Rows();i++){
        row_offsets[i+1] += row_offsets[i];
        for(int j=0;j<dense_matrix.Columns();j++){
            if(dense_matrix.Values()[j*dense_matrix.Rows()+i]!=0.0f){
                ++non_zeros;
                row_offsets[i+1]+=1;
            }
        }
    }
    std::vector<int> column_indices(non_zeros);
    std::vector<float> values(non_zeros);
    int num=0;
    float temp = 0.0f;
    for(int i=0;i<dense_matrix.Rows();i++){
        for(int j=0;j<dense_matrix.Columns();j++){
            temp = dense_matrix.Values()[j*dense_matrix.Rows()+i];
            if(temp!=0.0f){
                values[num] = temp;
                column_indices[num++] = j;
            }
        }
    }
    InitFromCSRFormat(dense_matrix.Rows(), dense_matrix.Columns(), non_zeros,
                      dense_matrix.BR(), dense_matrix.BC(), dense_matrix.Sparsity(),
                      row_offsets, column_indices, values);
}

CudaSparseMatrix::CudaSparseMatrix(const int &rows, const int &columns, const int &non_zeros,
                                   const int &br, const int &bc, const float &sparsity, const std::vector<int> &row_offsets,
                                   const std::vector<int> &column_indices, const std::vector<float> &values) {
    HostSparseMatrix hostSparseMatrix(rows, columns, non_zeros, br, bc, sparsity, row_offsets, column_indices, values);
    InitFromHostSparseMatrix(hostSparseMatrix);
}

CudaSparseMatrix::CudaSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                                   const float &sparsity, const std::vector<std::vector<float>> &values) {
    DenseMatrix dense_matrix(rows, columns, br, bc, sparsity, values);
    InitFromDenseMatrix(dense_matrix);
}

CudaSparseMatrix::CudaSparseMatrix(const DenseMatrix &dense_matrix) {
    InitFromDenseMatrix(dense_matrix);
}

CudaSparseMatrix::CudaSparseMatrix(const HostSparseMatrix &host_sparse_matrix) {
    InitFromHostSparseMatrix(host_sparse_matrix);
}

void CudaSparseMatrix::InitFromHostSparseMatrix(const HostSparseMatrix &host_sparse_matrix) {
    rows_ = host_sparse_matrix.Rows();
    columns_ = host_sparse_matrix.Columns();
    non_zeros_ = host_sparse_matrix.Non_zeros();
    br_ = host_sparse_matrix.BR();
    bc_ = host_sparse_matrix.BC();
    sparsity_ = host_sparse_matrix.Sparsity();

    CHECK_CUDA(cudaMalloc(&values_, sizeof(float) * non_zeros_))
    CHECK_CUDA(cudaMalloc(&column_indices_, sizeof(int) * non_zeros_))
    CHECK_CUDA(cudaMalloc(&row_offsets_, sizeof(int) * (rows_ + 1)))
    CHECK_CUDA(cudaMalloc(&row_indices_, sizeof(int) * rows_))

    CHECK_CUDA(cudaMemcpy(values_, host_sparse_matrix.Values(),
                          sizeof(float) * non_zeros_, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(column_indices_, host_sparse_matrix.ColumnIndices(),
                          sizeof(int) * non_zeros_, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(row_offsets_, host_sparse_matrix.RowOffsets(),
                          sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(row_indices_, host_sparse_matrix.RowIndices(),
                          sizeof(int) * rows_, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaStreamSynchronize(nullptr))
}

void CudaSparseMatrix::InitFromDenseMatrix(const DenseMatrix &dense_matrix) {
    HostSparseMatrix hostSparseMatrix(dense_matrix);
    InitFromHostSparseMatrix(hostSparseMatrix);
}