//
// Created by root on 4/26/22.
//

#ifndef SPUTNIK_CUDA_UTILS_H
#define SPUTNIK_CUDA_UTILS_H

#ifndef THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_
#define THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sputnik {

    typedef __half half;
    typedef __half2 half2;

    struct __align__(8) half4 {
    half2 x, y;
};

struct __align__(16) half8 {
half2 x, y, z, w;
};

struct __align__(8) short4 {
short2 x, y;
};

struct __align__(16) short8 {
short2 x, y, z, w;
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_
#endif //SPUTNIK_CUDA_UTILS_H
