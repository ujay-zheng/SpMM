#ifndef MATRIX_H
#define MATRIX_H

//#include "type_utils.h"

#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>
#include <type_utils.h>

#define CHECK_EQ(status, cudaSUCCESS) \
    if((status)!=(cudaSuccess))       \
        throw "not success";

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess); \
  } while (0)


/**
* @brief Create a row swizzle that maps thread blocks to rows in order of decreasing size.
* As shown in paper, it is a way to get a better performace with load balance
*/
void DecreasedSortedRowSwizzle(int rows, const int* row_offsets, int* row_indices);


/**
* @brief Simple dense-matrix class for managing pointers and memory allocation/deallocation
*/
class DenseMatrix{
protected:
    int rows_, columns_, total_num_;
    float *data_;

    DenseMatrix():rows_(0), columns_(0), data_(nullptr){};
public:
    /**
    * @brief Create a dense matrix with the specified properties from user-given vector
    *
    * @param rows the number of rows in the dense matrix.
    * @param columns the number of columns in the dense matrix
    * @param data the values of the dense matrix in row-major order
    */
    DenseMatrix(int rows, int columns, float *data);
    ~DenseMatrix(){
        delete[] data_;
    }

    DenseMatrix(const DenseMatrix&) = delete;
    DenseMatrix& operator=(const DenseMatrix&) = delete;
    DenseMatrix(DenseMatrix&&) = delete;
    DenseMatrix& operator=(DenseMatrix&&) = delete;

    const float* Data() const {return data_;}
    float* Data() {return data_;}

    int Rows() const {return rows_;}
    int Columns() const {return columns_;}
    int TotalNum() const {return total_num_;}

    friend void CSR(const DenseMatrix& dense_matrix, int &nonzeros, std::vector<int>& row_offsets,
                    std::vector<int>& column_indices, std::vector<int>& values);
};


template <typename Value>
class CudaSparseMatrix;


class SparseMatrix {
public:
    /**
    * @brief Create a sparse matrix using CSR format source data.
    *
    * @param rows The number of rows in the matrix.
    * @param columns The number of columns in the matrix.
    * @param nonzeros The number of nonzero values in the matrix.
    * @param row_offsets CSR row_offsets
    * @param row_indices row swizzle in decreased order
    * @param column_indices CSR column_indices
    * @param values CSR values
    * @param pad_to_rows Each row in the sparse matrix will be padded to a
    * multiple of this value. Defaults to 4, which enables the user of
    * 4-element vector loads and stores. For best performance, pad to
    * `kBlockItemsK`.
    */
    SparseMatrix(int rows, int columns, int nonzeros,
                 const std::vector<int>& row_offsets,
                 const std::vector<int>& column_indices,
                 const std::vector<float>& values,
                 int pad_rows_to = 4);

    /**
    * @brief Construct a sparse matrix from a CUDA sparse matrix.
    */
//    explicit SparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);

    /**
    * @brief Construct a sparse matrix from a dense matrix.


    /**
    * @brief Cleanup the underlying storage.
    */
    ~SparseMatrix() {
        delete[] values_;
        delete[] row_offsets_;
        delete[] column_indices_;
        delete[] row_indices_;
    }

    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    SparseMatrix(SparseMatrix&&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = delete;

    const float* Values() const { return values_; }
    float* Values() { return values_; }

    const int* RowOffsets() const { return row_offsets_; }
    int* RowOffsets() { return row_offsets_; }

    const int* ColumnIndices() const { return column_indices_; }
    int* ColumnIndices() { return column_indices_; }

    const int* RowIndices() const { return row_indices_; }
    int* RowIndices() { return row_indices_; }

    int Rows() const { return rows_; }

    int Columns() const { return columns_; }

    int NonZeros() const { return nonzeros_; }

    int PadRowsTo() const { return pad_rows_to_; }

    int NumElementsWithPadding() const { return num_elements_with_padding_; }

protected:
    SparseMatrix() : values_(nullptr),
                     row_offsets_(nullptr),
                     column_indices_(nullptr),
                     row_indices_(nullptr),
                     rows_(0),
                     columns_(0),
                     nonzeros_(0),
                     pad_rows_to_(0),
                     num_elements_with_padding_(0){}

    // Matrix value and index storage.
    float* values_;
    int* row_offsets_;
    int* column_indices_;

    // Swizzled row indices for load balancing.
    int* row_indices_;

    // Matrix meta-data.
    int rows_, columns_, nonzeros_;
    int pad_rows_to_, num_elements_with_padding_;

//    void InitFromCudaSparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);
    void InitFromDenseMatrix(const DenseMatrix& dense_matrix);
};


///**
// * @brief Simple gpu sparse-matrix class to for managing pointers and
// * memory allocation/deallocation.
// */
//template <typename Value>
//class CudaSparseMatrix {
//public:
//    /**
//    * @brief Create a sparse matrix using CSR format source data.
//    *
//    * @param rows The number of rows in the matrix.
//    * @param columns The number of columns in the matrix.
//    * @param nonzeros The number of nonzero values in the matrix.
//    * @param row_offsets CSR row_offsets
//    * @param row_indices row swizzle in decreased order
//    * @param column_indices CSR column_indices
//    * @param values CSR values
//    * @param pad_to_rows Each row in the sparse matrix will be padded to a
//    * multiple of this value. Defaults to 4, which enables the user of
//    * 4-element vector loads and stores. For best performance, pad to
//    * `kBlockItemsK`.
//    */
//
//    CudaSparseMatrix(int rows, int columns, int nonzeros,
//                     const std::vector<int>& row_offsets,
//                     const std::vector<int>& row_indices,
//                     const std::vector<int>& column_indices,
//                     const std::vector<float>& values,
//                     int pad_rows_to = 4);
//    /**
//    * @brief Construct a CUDA sparse matrix from a host sparse matrix.
//    */
//
//    explicit CudaSparseMatrix(const SparseMatrix& sparse_matrix);
//    /**
//    * @brief Cleanup the underlying storage.
//    */
//
//    ~CudaSparseMatrix() {
//        CUDA_CALL(cudaFree(values_));
//        CUDA_CALL(cudaFree(row_offsets_));
//        CUDA_CALL(cudaFree(column_indices_));
//        CUDA_CALL(cudaFree(row_indices_));
//    }
//
//    CudaSparseMatrix(const CudaSparseMatrix&) = delete;
//    CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
//    CudaSparseMatrix(CudaSparseMatrix&&) = delete;
//    CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;
//
//    // Datatype for indices in this matrix.
//    typedef typename Value2Index<Value>::Index Index;
//
//    const Value* Values() const { return values_; }
//    Value* Values() { return values_; }
//
//    const int* RowOffsets() const { return row_offsets_; }
//    int* RowOffsets() { return row_offsets_; }
//
//    const Index* ColumnIndices() const { return column_indices_; }
//    Index* ColumnIndices() { return column_indices_; }
//
//    const int* RowIndices() const { return row_indices_; }
//    int* RowIndices() { return row_indices_; }
//
//    int Rows() const { return rows_; }
//
//    int Columns() const { return columns_; }
//
//    int Nonzeros() const { return nonzeros_; }
//
//    int PadRowsTo() const { return pad_rows_to_; }
//
//    int NumElementsWithPadding() const { return num_elements_with_padding_; }
//protected:
//    CudaSparseMatrix() : values_(nullptr),
//                         row_offsets_(nullptr),
//                         column_indices_(nullptr),
//                         row_indices_(nullptr),
//                         rows_(0),
//                         columns_(0),
//                         nonzeros_(0),
//                         pad_rows_to_(0),
//                         num_elements_with_padding_(0){}
//
//    // Matrix value and index storage.
//    Value* values_;
//    int* row_offsets_;
//    Index* column_indices_;
//
//    // Swizzled row indices for load balancing.
//    int* row_indices_;
//
//    // Matrix meta-data.
//    int rows_, columns_, nonzeros_;
//    int pad_rows_to_, num_elements_with_padding_;
//
//    void InitFromSparseMatrix(const SparseMatrix& sparse_matrix);
//};


#endif