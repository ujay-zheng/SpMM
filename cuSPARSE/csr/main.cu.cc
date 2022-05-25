 #include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "spmm/spmm.h"

int main(int argc, char *argv[]){
    if(argc != 4) {
        std::cout << "we need a guide file, an output path and algorithm!" << std::endl;
        return 0;
    }
    char *guide = argv[1];
    char *out_csv = argv[2];
    char *algorithm = argv[3];
    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG1;
    if(algorithm[0] == '1'){
        alg = CUSPARSE_SPMM_CSR_ALG1;
    }else if(algorithm[0] == '2'){
        alg = CUSPARSE_SPMM_CSR_ALG2;
    }else if(algorithm[0] == '3'){
        alg = CUSPARSE_SPMM_CSR_ALG3;
    }else {
        std::cerr << "unknown algorithm!" << std::endl;
    }
    pipeline(guide, out_csv, alg);
    return 0;
}