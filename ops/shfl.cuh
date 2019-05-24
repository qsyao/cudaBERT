#ifndef CUDA_BERT_SHFL
#define CUDA_BERT_SHFL

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, srcLane, width);
#else
    return __shfl_down(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, srcLane, width);
#else
    return __shfl_xor(value, srcLane, width);
#endif
}

#endif