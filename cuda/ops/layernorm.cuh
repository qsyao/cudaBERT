#ifndef LAYERNORM_BERT_CUDA
#define LAYERNORM_BERT_CUDA

#include "shfl.cuh"
#include "../utils/common.h"
#include "../utils/manager.cuh"

template<typename T, typename U> 
void HostApplyLayerNorm(
    global_manager *handle,
    T* output,
    T* input,
    size_t n1,
    size_t n2,
    double epsilon,
    const T* gamma,
    const T* beta,
    T* merge_add = nullptr
    );
    
#endif
