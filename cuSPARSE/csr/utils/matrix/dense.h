#ifndef CU_SPARSE_CSR_DENSE_H
#define CU_SPARSE_CSR_DENSE_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

#include "cuda_utils.h"

class CudaMatrix;

class Matrix{
public:
    Matrix(const int &rows, const int &columns, float *values, const cusparseOrder_t &layout=CUSPARSE_ORDER_ROW);

    explicit Matrix(const CudaMatrix &cuda_dense);

    ~Matrix() { delete[] values_;}

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;
    Matrix& operator=(Matrix&&) = delete;

    cusparseOrder_t Layout() const {return layout_;}
    int LD() const {return ld_;}
    int TotalNum() const {return total_num_;}
    int Rows() const {return rows_;}
    int Columns() const {return columns_;}
    const float* Values() const {return values_;}
    float* Values() {return values_;}

protected:
    int rows_, columns_;
    float *values_;
    int  ld_;
    cusparseOrder_t layout_;
    int total_num_;
    Matrix():rows_(0), columns_(0), values_(nullptr), ld_(0), layout_(CUSPARSE_ORDER_ROW), total_num_(0){};
    Matrix(int row, int column, cusparseOrder_t layout)
    : rows_(row), columns_(column), values_(nullptr), ld_(0), layout_(layout), total_num_(0){};
};


class CudaMatrix: public Matrix{
public:

    CudaMatrix(const int &rows, const int &columns, float *values, const cusparseOrder_t &layout=CUSPARSE_ORDER_ROW);
    explicit CudaMatrix(const Matrix& host_matrix);

    ~CudaMatrix(){
        CHECK_CUSPARSE( cusparseDestroyDnMat(dense_matrix_description_) )
        CHECK_CUDA(cudaFree(values_))
        values_ = nullptr;  // avoid double free
    }

    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;
    CudaMatrix(CudaMatrix&&) = delete;
    CudaMatrix& operator=(CudaMatrix&&) = delete;

    cusparseDnMatDescr_t DenseMatrixDescription() const {return dense_matrix_description_;}

protected:
    cusparseDnMatDescr_t dense_matrix_description_;
    CudaMatrix():Matrix(), dense_matrix_description_(nullptr){};
    void InitFromHost(const Matrix &host_matrix);
};


#endif //CU_SPARSE_CSR_DENSE_H
