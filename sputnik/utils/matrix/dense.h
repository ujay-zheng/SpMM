#ifndef SPUTNIK_DENSE_H
#define SPUTNIK_DENSE_H

#include <vector>
#include <cuda_runtime.h>

#include "matrix/matrix.h"

class CudaDenseMatrix;

/**
 * @brief Class for managing the pointers and memory allocation/deallocation of a dense column-major matrix on host
 * */
class DenseMatrix:public Matrix{
public:
    /**
     * @brief create a dense column-major matrix on host from specified properties and user-given 2D vector
     * @param rows row number of matrix
     * @param columns column number of matrix
     * @param br block rows
     * @param bc block columns
     * @param sparsity sparsity
     * @param values matrix value in 2D vector
     * */
    DenseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                const float &sparsity, const std::vector<std::vector<float> > &values);
    /**
     * @brief convert matrix data from device to host
     * @param cudaDenseMatrix DenseMatrix on device
     * */
    explicit DenseMatrix(const CudaDenseMatrix &cuda_dense_matrix);
//    explicit DenseMatrix(const SparseMatrix& sparseMatrix);

    /**
     * @brief destructor for dense matrix on host
     * */
    ~DenseMatrix() override{ delete[] values_; }

    DenseMatrix(const DenseMatrix&) = delete;
    DenseMatrix& operator=(const DenseMatrix&) = delete;
    DenseMatrix(DenseMatrix&&) = delete;
    DenseMatrix& operator=(DenseMatrix&&) = delete;

protected:
    DenseMatrix():Matrix(){};
    DenseMatrix(const int &rows, const int &columns, const int &br, const int &bc, const float &sparsity)
                : Matrix(rows, columns, br, bc, sparsity) {};
};


/**
 * @brief Class for managing the pointers and memory allocation/deallocation of a dense column-major matrix on device
 * */
class CudaDenseMatrix:public Matrix{
public:
    /**
     * @brief create a dense column-major matrix on device from specified properties and user-given 2D vector on host
     * @param rows row number of matrix
     * @param br block rows
     * @param bc block columns
     * @param sparsity sparsity
     * @param columns column number of matrix
     * @param values matrix value in 2D vector
     * */
    CudaDenseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                    const float &sparsity, const std::vector<std::vector<float> > &values);
    /**
     * @brief convert matrix data from host to device
     * @param cudaDenseMatrix DenseMatrix on host
     * */
    explicit CudaDenseMatrix(const DenseMatrix& dense_matrix);

    /**
     * @brief destructor for dense matrix on device
     * */
    ~CudaDenseMatrix() override{
        CHECK_CUDA(cudaFree(values_))
    }

    CudaDenseMatrix(const CudaDenseMatrix&) = delete;
    CudaDenseMatrix& operator=(const CudaDenseMatrix&) = delete;
    CudaDenseMatrix(CudaDenseMatrix&&) = delete;
    CudaDenseMatrix& operator=(CudaDenseMatrix&&) = delete;

protected:
    CudaDenseMatrix():Matrix(){};
    CudaDenseMatrix(const int &rows, const int &columns, const int &br, const int &bc, const float &sparsity)
                    : Matrix(rows, columns, br, bc, sparsity) {};
    void InitFromDenseMatrix(const DenseMatrix& dense_matrix);
};


#endif //SPUTNIK_DENSE_H
