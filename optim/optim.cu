#include "optim.cuh"

__global__ void cu_apply_sgd_running_time(float* input, float *grad, size_t n, float lr) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        input[i] -= lr * grad[i];
    __syncthreads();
}

void apply_sgd_running_time(float* input,float* grad, size_t n, global_handle *handle) {
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long)65535, n/2014 + 1), 1, 1);
    cu_apply_sgd_running_time<<<blocks, threads, 0, handle->cal_stream>>>(input, grad, n, handle->learning_rate);
}
