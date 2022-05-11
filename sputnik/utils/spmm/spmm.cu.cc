#include <fstream>
#include <vector>
#include <sputnik/sputnik.h>

#include "spmm.h"

void cuda_construct_from_mtx(const std::string &mtx, CudaSparseMatrix **cuda_sparse_matrix,
                             CudaDenseMatrix **cuda_dense_matrix){
    std::ifstream in;
    in.open(mtx, std::ios::in);
    if(!in.is_open()){
        std::cerr << "open matrix file error!" << std::endl;
        exit(-1);
    }
    int M = 0, K = 0, N = 0, br = 0, bc = 0;
    float sparsity = 0.0f;
    in >> M >> K >> N >> br >> bc >> sparsity;

    std::vector<std::vector<float> > sparse_meta_data(M, std::vector<float>(K));
    std::vector<std::vector<float> > dense_meta_data(K, std::vector<float>(N));
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            in >> sparse_meta_data[i][j];
        }
    }
    for(int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            in >> dense_meta_data[i][j];
        }
    }

    *cuda_sparse_matrix = new CudaSparseMatrix(M, K, br, bc, sparsity, sparse_meta_data);
    *cuda_dense_matrix = new CudaDenseMatrix(K, N, br, bc, sparsity, dense_meta_data);
}


float spmm_timer(CudaSparseMatrix *cuda_sparse_matrix, CudaDenseMatrix *cuda_dense_matrix){
    float *output_matrix;
    float elapsedTime;
    CHECK_CUDA(cudaMalloc(&output_matrix,
                          sizeof(float) *  (cuda_sparse_matrix->Rows()) * (cuda_dense_matrix->Columns()) ))
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    CHECK_CUDA(sputnik::CudaSpmmBiasRelu(cuda_sparse_matrix->Rows(), cuda_sparse_matrix->Columns(),
                                        cuda_dense_matrix->Columns(), cuda_sparse_matrix->Non_zeros(),
                                        cuda_sparse_matrix->RowIndices(), cuda_sparse_matrix->Values(),
                                        cuda_sparse_matrix->RowOffsets(), cuda_sparse_matrix->ColumnIndices(),
                                        cuda_dense_matrix->Values(), nullptr, output_matrix, nullptr))
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(output_matrix))
    return elapsedTime;
}

void pipeline(const std::string &guide, const std::string &out_csv){
    std::ifstream in;
    std::ofstream out;
    in.open(guide, std::ios::in);
    if (!in.is_open()){
        std::cerr << "open guide file error!" << std::endl;
        exit(-1);
    }
    out.open(out_csv, std::ios::out);
    if (!out.is_open()){
        std::cerr << "open out csv error!" << std::endl;
        exit(-1);
    }
    out <<"M"<<","<<"K"<<","<<"N"<<","<<"block_size"<<","<<"sparsity"<<","<<"avg_latency(/ms)"<<std::endl;

    std::string mtx;
    CudaSparseMatrix *cuda_sparse_matrix = nullptr;
    CudaDenseMatrix *cuda_dense_matrix = nullptr;
    while(in >> mtx){
        cuda_construct_from_mtx(mtx, &cuda_sparse_matrix, &cuda_dense_matrix);
        if(cuda_sparse_matrix == nullptr || cuda_dense_matrix == nullptr) {
            std::cerr << "cuda_sparse_matrix and cuda_dense_matrix can not be nullptr!!" << std::endl;
            exit(-1);
        }
        float t = 0.0f, avg_t = 0.0f;
        int warmup = 10, repeat = 100;
        for(int i = 0;i < warmup; i++){
            spmm_timer(cuda_sparse_matrix, cuda_dense_matrix);
        }

        for(int i=0; i<repeat; i++){
            t += spmm_timer(cuda_sparse_matrix, cuda_dense_matrix);
        }
        avg_t = t / repeat;
        out << cuda_sparse_matrix->Rows() << "," << cuda_sparse_matrix->Columns() << "," << cuda_dense_matrix->Columns()
                << "," << cuda_sparse_matrix->BR() << "," << cuda_sparse_matrix->Sparsity() << "," << avg_t << std::endl;
        delete cuda_sparse_matrix;
        cuda_sparse_matrix = nullptr;
        delete cuda_dense_matrix;
        cuda_dense_matrix = nullptr;
    }
}