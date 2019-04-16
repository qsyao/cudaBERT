#ifndef CUDA_BERT_MATMUL
#define CUDA_BERT_MATMUL

#include <assert.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <assert.h>

#include "../utils/common.h"
#include "../utils/manager.cuh"

void deviceMatmul(cublasHandle_t handle, float *d_A, 
                std::vector<size_t> a_shape, 
                float *d_B,
                std::vector<size_t> b_shape, 
                float *d_C, 
                bool transpose_a_ = false, 
                bool transpose_b_ = false, 
                const float alpha = 1.0f, 
                const float beta = 0.0f,
                const int batchCount = -1, 
                const long long int strideA = 0, 
                const long long int strideB = 0, 
                const long long int strideC = 0);

void matmul(cublasHandle_t handle, 
            float *d_A, 
            std::vector<size_t> a_shape, 
            float *d_B,
            std::vector<size_t> b_shape, 
            float *d_C, 
            std::vector<size_t> c_shape, 
            bool transpose_a_ = false, 
            bool transpose_b_ = false, 
            const float alpha = 1.0f, 
            const float beta = 0.0f, 
            long long int custom_strideA = -1);

#endif
/*
int main()
{
    test();
}
*/
// nvcc matmul.cu -o matmul -lcublas -I /usr/local/cuda-9.0/samples/common/inc/ --std=c++11

