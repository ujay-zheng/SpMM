#include "matrix/dense.h"


void CudaMatrix::InitFromHost(const Matrix &host_matrix){
    rows_ = host_matrix.Rows();
    columns_ = host_matrix.Columns();
    layout_ = host_matrix.Layout();
    ld_ = host_matrix.LD();
    total_num_ = host_matrix.TotalNum();
    CHECK_CUDA( cudaMalloc((void**) &values_, host_matrix.TotalNum() * sizeof(float)))
    CHECK_CUDA(cudaMemcpy(values_, host_matrix.Values(), sizeof(float)*(host_matrix.TotalNum()), cudaMemcpyHostToDevice))
    CHECK_CUSPARSE( cusparseCreateDnMat(&dense_matrix_description_, rows_, columns_,
                                        host_matrix.LD(), values_,
                                        CUDA_R_32F, layout_))
}


Matrix::Matrix(const int &rows, const int &columns, float *values, const cusparseOrder_t &layout)
    :rows_(rows), columns_(columns), layout_(layout){
    if(layout_ == CUSPARSE_ORDER_ROW){
        ld_ = columns_;
        total_num_ = ld_ * rows_;
    }else if(layout_ == CUSPARSE_ORDER_COL){
        ld_ = rows_;
        total_num_ = ld_ * columns_;
    }else{
        std::cerr << "the value of layout is not a cusparseOrder_t type value!" << std::endl;
        exit(-1);
    }
    values_ = new float[total_num_];
    memcpy(values_, values, sizeof(float)*total_num_);
}

Matrix::Matrix(const CudaMatrix &cuda_dense):
                rows_(cuda_dense.Rows()), columns_(cuda_dense.Columns()), total_num_(cuda_dense.TotalNum()),
                ld_(cuda_dense.LD()), layout_(cuda_dense.Layout()){
    values_ = new float[total_num_];

    CHECK_CUDA(cudaMemcpy(values_, cuda_dense.Values(), sizeof(float)*total_num_, cudaMemcpyDeviceToHost))
}

CudaMatrix::CudaMatrix(const int &rows, const int &columns, float *values, const cusparseOrder_t &layout){
    Matrix host_matrix(rows, columns, values, layout);
    InitFromHost(host_matrix);
}

CudaMatrix::CudaMatrix(const Matrix &host_matrix){
    InitFromHost(host_matrix);
}