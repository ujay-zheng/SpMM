#include "SputnikConfig.h"

#include <iostream>

#include "spmm/spmm_utils.h"

int main(int argc, char *argv[]){
    if(argc != 3) {
        std::cout << "please give a guide file and out path!" << std::endl;
        return 0;
    }
    char *guide = argv[1];
    char *out_csv = argv[2];
    pipeline(guide, out_csv);
    return 0;
}
