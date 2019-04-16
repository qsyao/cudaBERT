#ifndef SOFTMAX2_CUDA_BERT
#define SOFTMAX2_CUDA_BERT

#include "shfl.cuh"
#include "../utils/common.h"
#include "../utils/manager.cuh"

template<typename T> 
void HostApplySoftmax(
    global_manager *handle,
    T* tensor,
    size_t n1,
    size_t n2
    );

#endif
