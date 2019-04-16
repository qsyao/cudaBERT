#ifndef LINEAR_CUDA_BERT
#define LINEAR_CUDA_BERT

#include <cuda_runtime.h>

#include "matmul.cuh"
#include "../utils/manager.cuh"

template <typename T> 
void Linear (global_manager *handler, 
        T* output, 
        T* input, 
        T* weights, 
        T* beta,
        size_t n, 
        size_t k, 
        size_t m,
        bool is_prepare=false,
        bool debug=false);

template <typename T>
void Batch_Linear (global_manager *handler, 
                   T* output, 
                   T* input, 
                   T* weights_0, 
                   T* beta_0,
                   T* weights_1, 
                   T* beta_1, 
                   T* weights_2, 
                   T* beta_2, 
                   size_t n, 
                   size_t k, 
                   size_t m,
                   bool is_prepare=false,
                   bool debug=false);

#endif
