#ifndef THIRD_PARTY_SPUTNIK_TYPE_UTILS_H_
#define THIRD_PARTY_SPUTNIK_TYPE_UTILS_H_

/**
 * @file @brief Defines utilities for working with mixes of data types
 * for storage and computation.
 */

#include "cuda_utils.h"

template <typename Value>
struct TypeUtils {
    static constexpr int kElementsPerScalar = 1;

    static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

    // The data type of our accumulators.
    typedef Value Accumulator;

    // The data type of a scalar value.
    typedef float ScalarValue;
};

template <>
struct TypeUtils<half2> {
    static constexpr int kElementsPerScalar = 2;

    static constexpr __device__ __forceinline__ bool IsMixed() { return true; }

    typedef float2 Accumulator;
    typedef half2 ScalarValue;
};


/**
 * @brief Functor to translate vector data types to vector index types.
 */
template <typename Value>
struct Value2Index {
    typedef int Index;
};

template <>
struct Value2Index<float2> {
    typedef int2 Index;
};

template <>
struct Value2Index<float4> {
    typedef int4 Index;
};

template <typename To, typename From>
__device__ __forceinline__ void Convert(const From *in, To *out) {
    // In the default case, don't perform any conversion. Reinterpret.
    *out = *reinterpret_cast<const To *>(in);
}

__device__ __forceinline__ void Convert(const float *in, half2 *out) {
    // Convert two 32-bit floats into 16-bit floats and pack into
    // a single half2.
    *out = __float22half2_rn(*reinterpret_cast<const float2 *>(in));
}

__device__ __forceinline__ void Convert(const float *in, half4 *out) {
    // Convert four 32-bit floats into 16-bit floats and pack into
    // a single half4.
    const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
    out->x = __float22half2_rn(in_f2[0]);
    out->y = __float22half2_rn(in_f2[1]);
}

__device__ __forceinline__ void Convert(const float *in, half8 *out) {
    // Convert 8 32-bit floats into 16-bits floats and pack into
    // a single half8
    const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
    out->x = __float22half2_rn(in_f2[0]);
    out->y = __float22half2_rn(in_f2[1]);
    out->z = __float22half2_rn(in_f2[2]);
    out->w = __float22half2_rn(in_f2[3]);
}

__device__ __forceinline__ void Convert(const short2 *x, int *out) {
    // Extract two 16-bit integers into 2 32-bit integers. Useful for
    // all variants of the kernels with low precision inputs. To
    // support a wide enough range of input matrix sizes, we need to
    // use 32-bits for all offsets derived from 16-bit indices.
    out[0] = static_cast<int>(x->x);
    out[1] = static_cast<int>(x->y);
}

__device__ __forceinline__ void Convert(const short4 *x, int *out) {
    Convert(&x->x, out);
    Convert(&x->y, out + 2);
}

__device__ __forceinline__ void Convert(const short2 x, int *out) {
    Convert(&x, out);
}

__device__ __forceinline__ void Convert(short4 x, int *out) {
    Convert(&x.x, out);
    Convert(&x.y, out + 2);
}

__device__ __forceinline__ void Convert(const half2 *x, float *out) {
    // Extract two 16-bit IEEE floating-point values into two 32-bit
    // IEEE floating-point values. Useful for pseudo-fp16 kernels.
    float2 tmp = __half22float2(*x);
    out[0] = tmp.x;
    out[1] = tmp.y;
}

#endif  // THIRD_PARTY_SPUTNIK_TYPE_UTILS_H_
