#ifndef SPUTNIK_MATRIX_H
#define SPUTNIK_MATRIX_H

#include <iostream>



#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cout << "CUDA API failed at line" << __LINE__ <<  "with error:" <<\
        cudaGetErrorString(status) <<  "(" << status << ")" << std::endl;      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

/**
 * @brief Base class for dense matrix whose data can be on host or device
 * */
class Matrix {
public:
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;
    Matrix& operator=(Matrix&&) = delete;

    /**
     * @brief destructor depends on whether data is on host or device
     * */
    virtual ~Matrix()= default;;

    int Rows() const {return rows_;}
    int Columns() const {return columns_;}
    int BR() const { return br_;}
    int BC() const { return bc_;}
    float Sparsity() const {return sparsity_;}
    const float* Values() const {return values_;}
    float* Values() {return values_;}

protected:
    int rows_, columns_, br_, bc_;
    float sparsity_;
    float *values_;
    Matrix():rows_(0), columns_(0), br_(0), bc_(0), sparsity_(0.0f), values_(nullptr){};
    Matrix(int rows, int columns, int br, int bc, float sparsity)
    :rows_(rows), columns_(columns), br_(br), bc_(bc), sparsity_(sparsity), values_(nullptr){};
};


/**
 * @brief Base class for sparse matrix whose data can be on host or device
 * */
class SparseMatrix: public Matrix{
public:
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    SparseMatrix(SparseMatrix&&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = delete;

    /**
     * @brief destructor depends on whether data is on host or device
     * */
    virtual ~SparseMatrix(){};

    const int* RowOffsets() const { return row_offsets_; }
    int* RowOffsets() { return row_offsets_; }

    const int* ColumnIndices() const { return column_indices_; }
    int* ColumnIndices() { return column_indices_; }

    const int* RowIndices() const { return row_indices_; }
    int* RowIndices() { return row_indices_; }

    int Non_zeros() const { return non_zeros_; }

protected:
    int* row_offsets_;
    int* column_indices_;

    int* row_indices_;

    int non_zeros_;

    SparseMatrix() : Matrix(0, 0, 1, 1, 0.0),
                     row_offsets_(nullptr),
                     column_indices_(nullptr),
                     row_indices_(nullptr),
                     non_zeros_(0){};
    SparseMatrix(int rows, int columns, int non_zeros, int br, int bc, float sparsity)
                    : Matrix(rows, columns, br, bc, sparsity),
                      row_offsets_(nullptr),
                      column_indices_(nullptr),
                      row_indices_(nullptr),
                      non_zeros_(non_zeros){};
};


#endif //SPUTNIK_MATRIX_H
