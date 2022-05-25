#include "spmm.h"

float latency_evaluate(CudaSparseMatrix *cuda_sparse, CudaMatrix *cuda_dense, cusparseSpMMAlg_t alg, int repeat){
    float *res = new float[cuda_sparse->Rows()*cuda_dense->Columns()];

    float *value = new float[cuda_sparse->Rows()*cuda_dense->Columns()];
    memset(value, 0, sizeof(float)*cuda_sparse->Rows()*cuda_dense->Columns());
    CudaMatrix result(cuda_sparse->Rows(), cuda_dense->Columns(), value, cuda_dense->Layout());

    float alpha = 1.0f;
    float beta = 0.0f;

    cusparseHandle_t handle = nullptr;
    void *dBuffer = nullptr;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle))
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cuda_sparse->SparseMatrixDescription(), cuda_dense->DenseMatrixDescription(),
            &beta, result.DenseMatrixDescription(), CUDA_R_32F,
            alg, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    int warmup=10;
    for(int i=0;i<warmup;i++){
        CHECK_CUSPARSE( cusparseSpMM(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, cuda_sparse->SparseMatrixDescription(), cuda_dense->DenseMatrixDescription(),
                &beta, result.DenseMatrixDescription(), CUDA_R_32F,
                alg, dBuffer) )
    }

    float avg_latency = 0.0f, elapsedTime=0.0f;
    for(int i=0;i<repeat;i++){
        elapsedTime=0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, nullptr);
        CHECK_CUSPARSE( cusparseSpMM(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, cuda_sparse->SparseMatrixDescription(), cuda_dense->DenseMatrixDescription(),
                &beta, result.DenseMatrixDescription(), CUDA_R_32F,
                alg, dBuffer) )
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        avg_latency += elapsedTime;
    }
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    avg_latency /= repeat;
    delete[] res;

    return avg_latency;
}

void cuda_construct_from_mtx(const std::string &mtx, CudaSparseMatrix **cuda_sparse,
                             CudaMatrix **cuda_dense, cusparseOrder_t layout){
    std::ifstream in;
    in.open(mtx, std::ios::in);
    if(!in.is_open()){
        std::cerr << "open matrix file error!" << std::endl;
        exit(-1);
    }
    int M = 0, K = 0, N = 0, br = 0, bc = 0;
    float sparsity = 0.0f;
    in >> M >> K >> N >> br >> bc >> sparsity;

    int sparse_num = M * K, dense_num = K * N;
    float *sparse_meta_data = new float[sparse_num], *dense_meta_data = new float[dense_num];

    for(int i=0;i<sparse_num;i++){
        in >> sparse_meta_data[i];
    }

    if(layout == CUSPARSE_ORDER_ROW){
        for(int i=0;i<dense_num;i++){
            in >> dense_meta_data[i];
        }
    }else if(layout == CUSPARSE_ORDER_COL){
        for(int i=0; i<K; i++){
            for(int j=0; j<N; j++){
                in >> dense_meta_data[j*K+i];
            }
        }
    }


    CudaMatrix dense_temp(M, K, sparse_meta_data, CUSPARSE_ORDER_ROW);
    *cuda_sparse = new CudaSparseMatrix(dense_temp, br, bc);
    *cuda_dense = new CudaMatrix(K, N, dense_meta_data, layout);
}

void pipeline(const std::string &guide, const std::string &out_csv, cusparseSpMMAlg_t alg){
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
    cusparseOrder_t layout = CUSPARSE_ORDER_ROW;
    if(alg == CUSPARSE_SPMM_CSR_ALG1){
        layout = CUSPARSE_ORDER_COL;
    }

    out <<"M"<<","<<"K"<<","<<"N"<<","<<"br"<<","<< "bc" <<","<<"sparsity"<<","<<"avg_latency(/ms)"<<std::endl;

    std::string mtx;
    CudaSparseMatrix *cuda_sparse = nullptr;
    CudaMatrix *cuda_dense = nullptr;
    float avg_t = 0.0f;
    while(in >> mtx){
        cuda_construct_from_mtx(mtx, &cuda_sparse, &cuda_dense, layout);
        if(cuda_sparse == nullptr || cuda_dense == nullptr) {
            std::cerr << "cuda_sparse_matrix and cuda_dense_matrix can not be nullptr!!" << std::endl;
            exit(-1);
        }
        avg_t = latency_evaluate(cuda_sparse, cuda_dense, alg);
        out << cuda_sparse->Rows() << "," << cuda_sparse->Columns() << "," << cuda_dense->Columns()
            << "," << cuda_sparse->Br()<< "," << cuda_sparse->Bc() << "," << cuda_sparse->Sparsity()
            << "," << avg_t << std::endl;
        delete cuda_sparse;
        cuda_sparse = nullptr;
        delete cuda_dense;
        cuda_dense = nullptr;
    }
}