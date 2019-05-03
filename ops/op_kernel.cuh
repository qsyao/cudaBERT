#ifndef CUDA_BERT_OP_KERNEL
#define CUDA_BERT_OP_KERNEL

#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../utils/manager.cuh"

class op_kernel {
  public:
    op_kernel() {}

    global_handle* handle;

    float* stored_input;
    float* stored_output;

public:
    op_kernel(global_handle* handle) 
                          : handle(handle) {}
};

#endif