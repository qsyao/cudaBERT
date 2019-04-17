#ifndef CROSSENTROPYLOSS_CUDA_BERT
#define CROSSENTROPYLOSS_CUDA_BERT

#include "../utils/common.h"
#include "../utils/manager.h"
#include "softmax.cu"

template <typename T, typename U>
__global__ void cuApplyCrossEntropyLoss(
    T *__restrict__ output_vals, T *__restrict__ input, U *__restrict__ classes, const int n1, const int n2)
{
    for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y)
    {
        T sum, max_;
        cuWelfordMax(input, n1, n2, i1, max_);
        T *vals = input + i1 * n2;
        const int blockx = blockDim.x * blockDim.y;
        const int threadx = threadIdx.x + threadIdx.y * blockDim.x;
        for (int i = threadx; i < n2; i += blockx)
        {
            vals[i] = exp(vals[i] - max_);
        }
        cuWelfordSum(input, n1, n2, i1, sum);
        const int label = static_cast<int>(classes[i1]);
        output_vals[i1] = -log(input[n2 * i1 + label]) + log(sum);
        __syncthreads();
    }
}

template <typename T>
__global__ void cuApplyCrossEntropyLossAverage(T *output, const int n1, const int n2)
{
    output[n1] = 0;
    __syncthreads();
    for (int index = blockIdx.x; index < n1; index += gridDim.x)
    {
        output[n1] += output[index];
    }
    if (blockIdx.x == 0)
        output[n1] /= n1;
}

template <typename T, typename U>
void HostApplyCrossEntropyLoss(
    global_manager *handle, T *output, T *input, U *classes,
    size_t n1, size_t n2)
{
    const dim3 threads(32, 1, 1);
    const dim3 blocks(1, min((long)65535, n1), 1);
    cuApplyCrossEntropyLoss<<<blocks, threads, 0, handle->get_cal_stream()>>>(
        output, input, classes, n1, n2);

    const dim3 threads1(min((long)32, n1), 1, 1);
    const dim3 blocks1(1, 1, 1);
    cuApplyCrossEntropyLossAverage<<<blocks1, threads1, 0, handle->get_cal_stream()>>>(
        output, n1, n2);
}

#endif