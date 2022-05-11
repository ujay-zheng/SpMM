#ifndef SPUTNIK_SPARSE_H
#define SPUTNIK_SPARSE_H

#include <vector>
#include "matrix/matrix.h"

class CudaSparseMatrix;

class HostSparseMatrix: public SparseMatrix{
public:
    /**
     * @brief Create a sparse matrix using CSR format source data.
     *
     * @param rows The number of rows in the matrix.
     * @param columns The number of columns in the matrix.
     * @param non_zeros The number of nonzero values in the matrix.
     * @param br block rows
     * @param bc block columns
     * @param sparsity sparsity
     * @param row_offsets CSR row_offsets
     * @param column_indices CSR column_indices
     * @param values CSR values
    */
    HostSparseMatrix(const int &rows, const int &columns, const int &non_zeros,
                 const int &br, const int &bc, const float &sparsity, const std::vector<int>& row_offsets,
                 const std::vector<int>& column_indices, const std::vector<float>& values);

    /**
     * @brief Construct a sparse matrix from a CUDA sparse matrix.
    */
    explicit HostSparseMatrix(const CudaSparseMatrix& cuda_sparse_matrix);

    /**
     * @brief Construct a sparse matrix from a dense matrix.
    */
    explicit HostSparseMatrix(const DenseMatrix& dense_matrix);

    /**
     * @brief Construct a sparse matrix from a dense matrix with user-given data.
    */
    HostSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                     const float &sparsity, const std::vector<std::vector<float> > &values);

    /**
     * @brief destructor for sparse matrix on host
    */
    ~HostSparseMatrix() override {
        delete[] values_;
        delete[] row_offsets_;
        delete[] column_indices_;
        delete[] row_indices_;
    }

    HostSparseMatrix(const HostSparseMatrix&) = delete;
    HostSparseMatrix& operator=(const HostSparseMatrix&) = delete;
    HostSparseMatrix(HostSparseMatrix&&) = delete;
    HostSparseMatrix& operator=(HostSparseMatrix&&) = delete;

protected:
    HostSparseMatrix() : SparseMatrix(){};
    /**
     * @brief get row_indices in descending order
    */
    void DecreasedSortedRowSwizzle();
    void InitFromCSRFormat(const int &rows, const int &columns, const int &non_zeros,
                           const int &br, const int &bc, const float &sparsity, const std::vector<int> &row_offsets,
                           const std::vector<int> &column_indices, const std::vector<float>& values);
    void InitFromDenseMatrix(const DenseMatrix &dense_matrix);
};


class CudaSparseMatrix: public SparseMatrix{
public:
    /**
      * @brief Create a sparse matrix on device
      *
      * @param rows The number of rows in the matrix.
      * @param columns The number of columns in the matrix.
      * @param non_zeros The number of nonzero values in the matrix.
      * @param br block rows
      * @param bc block columns
      * @param sparsity sparsity
      * @param row_offsets CSR row_offsets on device
      * @param column_indices CSR column_indices on device
      * @param values CSR values on device
     * */
    CudaSparseMatrix(const int &rows, const int &columns, const int &non_zeros,
                     const int &br, const int &bc, const float &sparsity, const std::vector<int> &row_offsets,
                     const std::vector<int> &column_indices, const std::vector<float> &values);
    explicit CudaSparseMatrix(const DenseMatrix &dense_matrix);
    /**
     * @brief Construct a CUDA sparse matrix from user-given meta data.
     */
    CudaSparseMatrix(const int &rows, const int &columns, const int &br, const int &bc,
                     const float &sparsity, const std::vector<std::vector<float> > &values);

    /**
     * @brief Construct a CUDA sparse matrix from a host sparse matrix.
     */
    explicit CudaSparseMatrix(const HostSparseMatrix &host_sparse_matrix);

    /**
    * @brief Cleanup the underlying storage.
    */
    ~CudaSparseMatrix() override {
        CHECK_CUDA(cudaFree(values_))
        CHECK_CUDA(cudaFree(row_offsets_))
        CHECK_CUDA(cudaFree(column_indices_))
        CHECK_CUDA(cudaFree(row_indices_))
    };
    CudaSparseMatrix(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix(CudaSparseMatrix&&) = delete;
    CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;

protected:
    CudaSparseMatrix() : SparseMatrix(){};
    void InitFromHostSparseMatrix(const HostSparseMatrix &host_sparse_matrix);
    void InitFromDenseMatrix(const DenseMatrix &dense_matrix);
};


#endif //SPUTNIK_SPARSE_H
