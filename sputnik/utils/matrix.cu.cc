#include "matirx.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

void DecreasedSortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
    // Create our unsorted row indices.
    std::vector<int> swizzle_staging(rows);
    std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

    // Argsort the row indices based on their length.
    std::sort(swizzle_staging.begin(), swizzle_staging.end(),
              [&row_offsets](int idx_a, int idx_b) {
                  int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
                  int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
                  return length_a > length_b;
              });

    // Copy the ordered row indices to the output.
    std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}


DenseMatrix::DenseMatrix(int rows, int columns, float *data){
    rows_ = rows;
    columns_ = columns;
    total_num_ = rows_ * columns_;
    data_ = new float[total_num_];
    std::memcpy(data_, data, total_num_*sizeof(float));
}

void PadSparseMatrix(const std::vector<int> &row_offsets,
                     const std::vector<float> &values,
                     const std::vector<int> &column_indices, int row_padding,
                     std::vector<int> *row_offsets_out,
                     std::vector<float> *values_out,
                     std::vector<int> *column_indices_out) {
    if(row_padding< 0)
        throw "Row padding factor must be greater than zero.";
    if (row_padding < 2) {
        row_offsets_out->assign(row_offsets.begin(), row_offsets.end());
        values_out->assign(values.begin(), values.end());
        column_indices_out->assign(column_indices.begin(), column_indices.end());
        return;
    }
    row_offsets_out->push_back(0);

    int offset = 0;
    for (int i = 0; i < row_offsets.size() - 1; ++i) {
        // Copy the existing values and column indices for this row to
        // the output.
        int row_length = row_offsets[i + 1] - row_offsets[i];
        values_out->resize(values_out->size() + row_length);
        column_indices_out->resize(column_indices_out->size() + row_length);
        std::copy(values.begin() + row_offsets[i],
                  values.begin() + row_offsets[i + 1],
                  values_out->begin() + offset);
        std::copy(column_indices.begin() + row_offsets[i],
                  column_indices.begin() + row_offsets[i + 1],
                  column_indices_out->begin() + offset);
        offset += row_length;

        // Calculate the number of zeros that need to be inserted in
        // this row to reach the desired padding factor.
        int residue = offset % row_padding;
        int to_add = (row_padding - residue) % row_padding;
        for (; to_add > 0; --to_add) {
            values_out->push_back(0.0);

            // NOTE: When we pad with zeros the column index that we assign
            // the phantom zero needs to be a valid column index s.t. we
            // don't index out-of-range into the dense rhs matrix when
            // computing spmm. Here we set all padding column-offsets to
            // the same column as the final non-padding weight in the row.
            column_indices_out->push_back(column_indices_out->back());
            ++offset;
        }
        row_offsets_out->push_back(offset);
    }
}


SparseMatrix::SparseMatrix(int rows, int columns, int nonzeros,
                           const std::vector<int>& row_offsets,
                           const std::vector<int>& column_indices,
                           const std::vector<float>& values,
                           int pad_rows_to)
        : rows_(rows),
          columns_(columns),
          nonzeros_(nonzeros),
          pad_rows_to_(pad_rows_to) {
    if(pad_rows_to_ > columns)
        throw "Rows cannot be padded to more values than there are columns.";

    // Pad the rows to the desired length.
    std::vector<int> row_offsets_staging, column_indices_staging;
    std::vector<float> values_staging;
    PadSparseMatrix(row_offsets, values, column_indices, pad_rows_to,
                    &row_offsets_staging, &values_staging,
                    &column_indices_staging);

    // Figure out exactly how much storage we need for the padded matrices,
    // allocate the storage, and copy the matrices into our storage.
    num_elements_with_padding_ = row_offsets_staging[rows_];

    values_ = new float[num_elements_with_padding_];
    column_indices_ = new int[num_elements_with_padding_];
    row_offsets_ = new int[rows_ + 1];

    // Copy the data into our allocated buffers.
    std::memcpy(values_, values_staging.data(),
                num_elements_with_padding_ * sizeof(float));
    std::memcpy(column_indices_, column_indices_staging.data(),
                num_elements_with_padding_ * sizeof(int));
    std::memcpy(row_offsets_, row_offsets_staging.data(),
                (rows_ + 1) * sizeof(int));

    // Allocate storage for our swizzled row indices and set the values.
    row_indices_ = new int[rows_];
    DecreasedSortedRowSwizzle(rows_, row_offsets_, row_indices_);
}

//SparseMatrix::SparseMatrix(const CudaSparseMatrix<float>& sparse_matrix) {
//    InitFromCudaSparseMatrix(sparse_matrix);
//}

//void SparseMatrix::InitFromCudaSparseMatrix(const CudaSparseMatrix<float> &sparse_matrix) {
//    // Copy the sparse matrix meta-data.
//    rows_ = sparse_matrix.Rows();
//    columns_ = sparse_matrix.Columns();
//    nonzeros_ = sparse_matrix.Nonzeros();
//    pad_rows_to_ = sparse_matrix.PadRowsTo();
//    num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
//
//    // Allocate memory on the CPU for our matrix.
//    values_ = new float[num_elements_with_padding_];
//    column_indices_ = new int[num_elements_with_padding_];
//    row_offsets_ = new int[rows_ + 1];
//    row_indices_ = new int[rows_];
//
//    // Copy the results to the CPU.
//    CUDA_CALL(cudaMemcpy(values_, sparse_matrix.Values(),
//                         sizeof(float) * num_elements_with_padding_,
//                         cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(column_indices_, sparse_matrix.ColumnIndices(),
//                         sizeof(int) * num_elements_with_padding_,
//                         cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(),
//                         sizeof(int) * (rows_ + 1), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(),
//                         sizeof(int) * rows_, cudaMemcpyDeviceToHost));
//}

void CSR(const DenseMatrix& dense_matrix, int &nonzeros,
                      std::vector<int>& row_offsets,
                      std::vector<int>& column_indices,
                      std::vector<int>& values){
    nonzeros = 0;
    row_offsets.resize(dense_matrix.rows_+1);
    row_offsets[0] = 0;
    for(int i=0;i<dense_matrix.rows_;i++){
        row_offsets[i+1] += row_offsets[i];
        for(int j=0;j<dense_matrix.columns_;j++){
            if(*(dense_matrix.data_+i*dense_matrix.columns_+j)!=0.0f){
                ++nonzeros;
                row_offsets[i+1]+=1;
            }
        }
    }
    values.resize(nonzeros);
    column_indices.resize(nonzeros);
    int num=0;
    for(int i=0;i<dense_matrix.rows_;i++){
        for(int j=0;j<dense_matrix.columns_;j++){
            float temp = *(dense_matrix.data_+i*dense_matrix.columns_+j);
            if(temp!=0.0f){
                values[num] = temp;
                column_indices[num++] = j;
            }
        }
    }
}


//template<typename Value>
//CudaSparseMatrix(int rows, int columns, int nonzeros,
//                 const std::vector<int>& row_offsets,
//                 const std::vector<int>& row_indices,
//                 const std::vector<int>& column_indices,
//                 const std::vector<float>& values,
//                 int pad_rows_to = 4){
//    if(pad_rows_to % TypeUtils<Value>::kElementsPerScalar != 0)
//        throw "The number of elements in each row must be divisible "
//              "by the number of elements per scalar value for the "
//              "specified data type.";
//    SparseMatrix sparse_matrix(rows, columns, nonzeros,
//                               row_offsets, row_indices, column_indices,
//                               values, pad_rows_to);
//    InitFromSparseMatrix(sparse_matrix);
//}

//template <typename Value>
//CudaSparseMatrix<Value>::CudaSparseMatrix(const SparseMatrix &sparse_matrix) {
//    // The number of nonzeros in each row must be divisible by the number of
//    // elements per scalar for the specified data type.
//    for (int i = 0; i < sparse_matrix.Rows(); ++i) {
//        int nnz = sparse_matrix.RowOffsets()[i + 1] - sparse_matrix.RowOffsets()[i];
//        if(nnz % TypeUtils<Value>::kElementsPerScalar != 0)
//            throw "The number of elements in each row must be divisible by"
//                  "the number of elements per scalar value for the specified"
//                  "data type.";
//    }
//    InitFromSparseMatrix(sparse_matrix);
//}

//template <typename Value>
//void CudaSparseMatrix<Value>::InitFromSparseMatrix(
//        const SparseMatrix &sparse_matrix) {
//    // Copy the sparse matrix meta-data.
//    rows_ = sparse_matrix.Rows();
//    columns_ = sparse_matrix.Columns();
//    nonzeros_ = sparse_matrix.Nonzeros();
//    pad_rows_to_ = sparse_matrix.PadRowsTo();
//    num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
//
//    // Allocate memory on the GPU for our matrix.
//    float *values_float = nullptr;
//    int *column_indices_int = nullptr;
//    CUDA_CALL(
//            cudaMalloc(&values_float, sizeof(float) * num_elements_with_padding_));
//    CUDA_CALL(cudaMalloc(&column_indices_int,
//                         sizeof(int) * num_elements_with_padding_));
//    CUDA_CALL(cudaMalloc(&row_offsets_, sizeof(int) * (rows_ + 1)));
//    CUDA_CALL(cudaMalloc(&row_indices_, sizeof(int) * rows_));
//
//    // Copy the results to the GPU.
//    CUDA_CALL(cudaMemcpy(values_float, sparse_matrix.Values(),
//                         sizeof(float) * num_elements_with_padding_,
//                         cudaMemcpyHostToDevice));
//    CUDA_CALL(cudaMemcpy(column_indices_int, sparse_matrix.ColumnIndices(),
//                         sizeof(int) * num_elements_with_padding_,
//                         cudaMemcpyHostToDevice));
//    CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(),
//                         sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice));
//    CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(),
//                         sizeof(int) * rows_, cudaMemcpyHostToDevice));
//    CUDA_CALL(cudaStreamSynchronize(nullptr));
//
//    // Allocate memory for the values and indices in the target datatype.
//    int elements =
//            num_elements_with_padding_ / TypeUtils<Value>::kElementsPerScalar;
//    CUDA_CALL(cudaMalloc(&values_, sizeof(Value) * elements));
//    CUDA_CALL(cudaMalloc(&column_indices_, sizeof(Index) * elements));
//
//    // Convert to the target datatype.
//    CUDA_CALL(Convert(values_float, values_, num_elements_with_padding_));
//    CUDA_CALL(
//            Convert(column_indices_int, column_indices_, num_elements_with_padding_));
//    CUDA_CALL(cudaStreamSynchronize(nullptr));
//
//    // Free the temporary memory.
//    CUDA_CALL(cudaFree(values_float));
//    CUDA_CALL(cudaFree(column_indices_int));
//}

//template class CudaSparseMatrix<float>;