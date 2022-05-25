#ifndef CU_SPARSE_CSR_SPARSE_H
#define CU_SPARSE_CSR_SPARSE_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstring>
#include <vector>

#include "matrix/dense.h"
#include "cuda_utils.h"

class CudaSparseMatrix;

class SparseMatrix{
public:

    SparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                 const int &non_zeros, int *row_offsets, int *column_indices, float *values);

    explicit SparseMatrix(const CudaSparseMatrix& cuda_sparse);

    ~SparseMatrix() {
        delete[] values_;
        delete[] row_offsets_;
        delete[] column_indices_;
    }

    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    SparseMatrix(SparseMatrix&&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = delete;

    int Rows() const {return rows_;}
    int Columns() const {return columns_;}
    int TotalNum() const {return total_num_;}
    int Br() const {return br_;}
    int Bc() const {return bc_;}
    int NonZeros() const {return non_zeros_;}
    float Sparsity() const {return sparsity_;}
    const float* Values() const {return values_;}
    float* Values() {return values_;}
    const int* RowOffsets() const {return row_offsets_;}
    int* RowOffsets() {return row_offsets_;}
    const int* ColumnIndices() const {return column_indices_;}
    int* ColumnIndices(){return column_indices_;}
protected:
    int rows_, columns_, total_num_, br_, bc_, non_zeros_;
    float sparsity_;
    float *values_;
    int *row_offsets_, *column_indices_;
    SparseMatrix(): rows_(0), columns_(0), total_num_(0), br_(0), bc_(0), non_zeros_(0),
                    values_(nullptr), row_offsets_(nullptr), sparsity_(0.0f),
                    column_indices_(nullptr){};
    SparseMatrix(const int &rows, const int &columns, const int &non_zeros, const int &br, const int &bc):
                rows_(rows), columns_(columns), non_zeros_(non_zeros), br_(br), bc_(bc), total_num_(rows*columns),
                sparsity_(static_cast<float>(rows*columns-non_zeros)/static_cast<float>(rows*columns)), values_(nullptr),
                row_offsets_(nullptr), column_indices_(nullptr){}
};


class CudaSparseMatrix: public SparseMatrix{
public:
    CudaSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                     const int &non_zeros, int *row_offsets, int *column_indices, float *values);

    explicit CudaSparseMatrix(const CudaMatrix &cuda_dense, int br, int bc);

    explicit CudaSparseMatrix(const SparseMatrix &host_sparse);

    ~CudaSparseMatrix() {
        CHECK_CUSPARSE( cusparseDestroySpMat(sparse_matrix_description_))
        CHECK_CUDA(cudaFree(values_))
        CHECK_CUDA(cudaFree(row_offsets_))
        CHECK_CUDA(cudaFree(column_indices_))
        // avoid double free
        values_ = nullptr;
        row_offsets_ = nullptr;
        column_indices_ = nullptr;
    };
    CudaSparseMatrix(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix(CudaSparseMatrix&&) = delete;
    CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;

    cusparseSpMatDescr_t SparseMatrixDescription() const {return sparse_matrix_description_;}

protected:
    cusparseSpMatDescr_t sparse_matrix_description_;
    CudaSparseMatrix(): SparseMatrix(){};
    CudaSparseMatrix(const int &rows, const int &columns, const int &non_zeros, const int &br, const int &bc):
                     SparseMatrix(rows, columns, non_zeros, br, bc){}
};

#endif //CU_SPARSE_CSR_SPARSE_H
