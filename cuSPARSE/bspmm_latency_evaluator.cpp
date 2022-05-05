#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdlib>           // EXIT_FAILURE
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "bspmm.h"
#include <cstring>

#define cuda_date_type CUDA_R_32F   //if data_type = __halt then cuda_data_type = CUDA_R_16F
#define data_type float

inline int matrix2row_major(int i, int j, int cols);
template<typename T>
inline void get_ell_mes(T *data, int &ell_col_num, std::vector<int> &non_zero_id, int num_rows, int num_cols, int block_size);
template<typename T>
inline int bspmm(T *hA, T *hB, T *hC, int *h_A_columns, T *h_A_values,
                 int A_num_rows, int A_num_cols, int B_num_cols, int ell_col_num,
                 int ell_blocksize, T alpha, T beta, cudaDataType cuda_val_type,
                 float &elapsedTime);
template<typename T>
inline float latency_test(int num_trials, int A_num_rows, int A_num_cols, int B_num_cols,
                          T *hA, T *hB, T *hC, int A_ell_blocksize);
template<typename T>
inline void evaluator(std::string guide_file, std::string out_file);

int main(int argc, char *argv[]){
    if(argc != 3) {
        std::cout << "please give a guide file and out path!" << std::endl;
        return 0;
    }
    char *source_addr = argv[1];
    char *out_file = argv[2];
    evaluator<data_type>(source_addr, out_file);
    return 0;
}


inline int matrix2row_major(int i, int j, int cols){
    return i*cols+j;
}

template<typename T>
inline void get_ell_mes(T *data, int &ell_col_num, std::vector<int> &non_zero_id, int num_rows, int num_cols, int block_size){
    int ell_row = num_rows / block_size, ell_col = num_cols/block_size;
    std::vector<std::vector<int> > id_matrix(ell_row, std::vector<int>(0));
    for(int i=0;i<ell_row;i++)
        for(int j=0;j<ell_col;j++)
            if(data[matrix2row_major(i*block_size, j*block_size, num_cols)] != 0.0f) id_matrix[i].push_back(j);
    ell_col_num = 0;
    for(int i=0;i<ell_row;i++) if(ell_col_num<id_matrix[i].size()) ell_col_num=id_matrix[i].size();
    for(int i=0;i<ell_row;i++){
        int j=0;
        for(;j<id_matrix[i].size();j++) non_zero_id.push_back(id_matrix[i][j]);
        for(;j<ell_col_num;j++) non_zero_id.push_back(-1);
    }
}



template<typename T>
inline int bspmm(T *hA, T *hB, T *hC, int *h_A_columns, T *h_A_values,
                 int A_num_rows, int A_num_cols, int B_num_cols, int ell_col_num,
                 int ell_blocksize, T alpha, T beta, cudaDataType cuda_val_type,
                 float &elapsedTime){
    int *d_A_columns;
    T *dA, *d_A_values, *dB, *dC;
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matA_dense, matB, matC;
    void* spmm_buffer    = NULL, *dense2sparse_buffer = NULL, *d_A_buffer = NULL;
    size_t bufferSize = 0;
    int sparse_val_num = ell_col_num * A_num_rows;
    int B_num_rows = A_num_cols;
    int lda = A_num_cols, ldb = B_num_cols, ldc = B_num_cols;
    int A_size = A_num_rows * A_num_cols, B_size = ldb * B_num_rows, C_size = ldc * A_num_rows;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    size_t T_size = sizeof(T), col_type_size = sizeof(int);
    DENSE2BLOCKSPARSE(handle, matA_dense, matA, dA, hA, h_A_columns, d_A_columns, h_A_values, d_A_values,
                      dense2sparse_buffer, A_num_rows, A_num_cols, lda, A_size, T_size, col_type_size, ell_col_num,
                      sparse_val_num, ell_blocksize, CUSPARSE_ORDER_ROW, cuda_val_type, CUSPARSE_INDEX_32I)
    CREATE_DENSE_DESCRIBE( matB,dB, hB, B_num_rows, B_num_cols, ldb, B_size, T_size, CUSPARSE_ORDER_ROW, cuda_val_type)
    CREATE_DENSE_DESCRIBE( matC,dC, hC, A_num_rows, B_num_cols, ldc, C_size, T_size, CUSPARSE_ORDER_ROW, cuda_val_type)
    SPMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, matB, beta, matC,
         cuda_val_type, CUSPARSE_SPMM_ALG_DEFAULT, spmm_buffer, elapsedTime)

    DESTORY_MATRIX_DESCRIBE(matA_dense, matA, matB, matC, handle)
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(T), cudaMemcpyDeviceToHost) )
//    float temp = 0.0;
//    for(int i=0;i<12;i++){
//        if(i%3==0) std::cout<<std::endl;
//        std::cout << hC[i] << " ";
//    }
//    std::cout << std::endl;
    FREE_CUDA_MEMORY(dense2sparse_buffer, spmm_buffer, d_A_columns, d_A_values, dA, dB, dC)
}

template<typename T>
inline float latency_test(int num_trials, int A_num_rows, int A_num_cols, int B_num_cols,
                          T *hA, T *hB, T *hC, int A_ell_blocksize){
    T *h_A_values;
    float alpha = 1.0f,beta = 0.0f;
    int A_ell_cols = 0;

    std::vector<int> h_A_columns;
    get_ell_mes(hA, A_ell_cols, h_A_columns, A_num_rows, A_num_cols, A_ell_blocksize);
    A_ell_cols*=A_ell_blocksize;
    h_A_values = new T[A_ell_cols*A_num_rows];
    int warm_up= 10;
    float elapsedTime=0.0, time_total = 0.0, time_avg=0.0;
    for(int i=0;i<warm_up;i++){
        bspmm<T>(hA, hB, hC, h_A_columns.data(), h_A_values, A_num_rows, A_num_cols, B_num_cols,
                 A_ell_cols, A_ell_blocksize, alpha, beta, cuda_date_type, elapsedTime);
    }
    for(int i=0;i<num_trials;i++){
        bspmm<T>(hA, hB, hC, h_A_columns.data(), h_A_values, A_num_rows, A_num_cols, B_num_cols,
                 A_ell_cols, A_ell_blocksize, alpha, beta, cuda_date_type, elapsedTime);
        time_total += elapsedTime;
    }
    time_avg = time_total / num_trials;
    delete []h_A_values;
    return time_avg;
}

template<typename T>
inline void evaluator(std::string guide_file, std::string out_file){
    std::ifstream guide, source;
    std::ofstream out;

    guide.open(guide_file, std::ios::in);
    if(!guide.is_open())
        throw "open guide file error!";
    out.open(out_file, std::ios::out);
    if(!out.is_open())
        throw "open out file error!";
    out <<"M"<<","<<"K"<<","<<"N"<<","<<"block_size"<<","<<"sparsity"<<","<<"avg_latency(/ms)"<<std::endl;

    int A_num_rows, A_num_cols, B_num_cols;
    double sparsity = 0.0;
    T *hA, *hB, *hC;
    int br, bc, A_ell_blocksize = 0;
    float elapsedTime = 0.0;
    std::string source_addr;

    while(guide >> source_addr){
        source.open(source_addr, std::ios::in);
        if(!source.is_open())
            throw "open source file error!";
        source >> A_num_rows >> A_num_cols >> B_num_cols >> br >> bc >> sparsity;
        if(br!=bc) {
            source.close();
            continue;
        }
        A_ell_blocksize = br;
        int total_A = A_num_rows * A_num_cols, total_B = A_num_cols * B_num_cols, total_C = A_num_rows*B_num_cols;
        hA = new T[total_A];
        hB = new T[total_B];
        hC = new T[total_C];
        float temp = 0.0f;
        for(int i=0;i<total_A;i++){
            source >> temp;
            *(hA+i) = temp;
        }
        for(int i=0;i<total_B;i++){
            source >> temp;
            *(hB+i) = temp;
        }
        elapsedTime = latency_test(100, A_num_rows, A_num_cols, B_num_cols, hA,  hB, hC,  A_ell_blocksize);
        out <<A_num_rows<<","<<A_num_cols<<","<<B_num_cols<<","<<A_ell_blocksize<<","<<sparsity<<","<<elapsedTime<<std::endl;

        source.close();

        delete []hA;
        delete []hB;
        delete []hC;
    }
    guide.close();
    out.close();
}
