#include "matrix/dense.h"
#include "matrix/sparse.h"

DenseMatrix::DenseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                         const float &sparsity, const std::vector<std::vector<float> > &values)
    : Matrix(rows, columns, br, bc, sparsity){
    values_ = new float[rows_*columns_];
    for(int j=0;j<columns;j++){
        for(int i=0;i<rows;i++){
            values_[j*rows_+i] = values[i][j];
        }
    }
}

DenseMatrix::DenseMatrix(const CudaDenseMatrix &cuda_dense_matrix)
    : Matrix(cuda_dense_matrix.Rows(), cuda_dense_matrix.Columns(),
             cuda_dense_matrix.BR(), cuda_dense_matrix.BC(), cuda_dense_matrix.Sparsity()){
    int total_num = rows_ * columns_;
    values_ = new float[total_num];
    CHECK_CUDA(cudaMemcpy(values_, cuda_dense_matrix.Values(),
                          sizeof(float) * total_num,
                          cudaMemcpyDeviceToHost));
}

CudaDenseMatrix::CudaDenseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                                 const float &sparsity, const std::vector<std::vector<float>> &values) {
    DenseMatrix denseMatrix(rows, columns, br, bc, sparsity, values);
    InitFromDenseMatrix(denseMatrix);
}

CudaDenseMatrix::CudaDenseMatrix(const DenseMatrix &dense_matrix) {
    InitFromDenseMatrix(dense_matrix);
}

void CudaDenseMatrix::InitFromDenseMatrix(const DenseMatrix &dense_matrix) {
    rows_ = dense_matrix.Rows();
    columns_ = dense_matrix.Columns();
    br_ = dense_matrix.BR();
    bc_ = dense_matrix.BC();
    sparsity_ = dense_matrix.Sparsity();

    int total_num = rows_ * columns_;
    CHECK_CUDA(cudaMalloc(&values_, sizeof(float) * total_num));
    CHECK_CUDA(cudaMemcpy(values_, dense_matrix.Values(),
                          sizeof(float) * total_num,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaStreamSynchronize(nullptr));
}