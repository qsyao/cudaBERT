#ifndef CROSSENTROPYLOSS_CUDA_BERT
#define CROSSENTROPYLOSS_CUDA_BERT

#include "../utils/common.h"
#include "../utils/manager.h"
#include "softmax.cu"

template <typename T, typename U>
__global__ void cuApplyCrossEntropyLoss(
    T *__restrict__ output_vals, T *__restrict__ input, U *__restrict__ classes, const int n1, const int n2, T *__restrict__ weight = nullptr)
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
        if (weight != nullptr)
        {
            output_vals[i1] = weight[label] * (-log(input[n2 * i1 + label]) + log(sum));
        }
        else
        {
            output_vals[i1] = -log(input[n2 * i1 + label]) + log(sum);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void cuApplyCrossEntropyLossAverage(T *output, const int n1, const int n2)
{
    T average_output = 0;
    for (int index = blockIdx.x; index < n1; index += gridDim.x)
    {
        average_output += output[index];
    }
    if (blockIdx.x == 0)
        output[n1] = average_output / n1;
}

template <typename T, typename U>
void HostApplyCrossEntropyLoss(
    global_manager *handle, T *output, T *input, U *classes,
    size_t n1, size_t n2, T *weight = nullptr)
{
    const dim3 threads(32, 1, 1);
    const dim3 blocks(1, min((long)65535, n1), 1);
    cuApplyCrossEntropyLoss<<<blocks, threads, 0, handle->get_cal_stream()>>>(
        output, input, classes, n1, n2, weight);

    const dim3 threads1(min((long)32, n1), 1, 1);
    const dim3 blocks1(1, 1, 1);
    cuApplyCrossEntropyLossAverage<<<blocks1, threads1, 0, handle->get_cal_stream()>>>(
        output, n1, n2);
}

template <typename T, typename U>
__global__ void cuApplyCrossEntropyLossGradient(
    T *__restrict__ dout, T *__restrict__ input, U *__restrict__ classes, const int n1, const int n2, T *grad_input, T *__restrict__ weight = nullptr)
{
    for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y)
    {
        T sum, max_;
        cuWelfordMax(input, n1, n2, i1, max_);
        T *vals = input + i1 * n2;
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        for (int i = thrx; i < n2; i += numx)
        {
            vals[i] = exp(vals[i] - max_);
        }
        cuWelfordSum(input, n1, n2, i1, sum);
        const int label = static_cast<int>(classes[i1]);
        T *k_grad_input = grad_input + i1 * n2;
        if (weight != nullptr)
        {
            for (int l = thrx; l < n2; l += numx)
            {
                T f_grad_input = 1.0 / n1 * vals[l] / sum * weight[label];
                if (l == label)
                {
                    f_grad_input -= 1.0 / n1 * weight[label];
                    k_grad_input[l] = f_grad_input * dout[i1];
                }
                else
                {
                    k_grad_input[l] = f_grad_input * dout[i1];
                }
            }
        }
        else
        {
            for (int l = thrx; l < n2; l += numx)
            {
                T f_grad_input = 1.0 / n1 * vals[l] / sum;
                if (l == label)
                {
                    f_grad_input -= 1.0 / n1;
                    k_grad_input[l] = f_grad_input * dout[i1];
                }
                else
                {
                    k_grad_input[l] = f_grad_input * dout[i1];
                }
            }
        }
        __syncthreads();
    }
}

template <typename T, typename U>
void HostApplyCrossEntropyLossGradient(
    global_manager *handle,
    T *dout,
    T *input,
    size_t n1, size_t n2, U *classes,
    T *grad_input, T *weight = nullptr)
{
    const dim3 threads(32, 1, 1);
    const dim3 blocks(1, min((long)65535, n1), 1);
    cuApplyCrossEntropyLossGradient<<<blocks, threads, 0, handle->get_cal_stream()>>>(
        dout, input, classes, n1, n2, grad_input, weight);
    debug_tensor_gpu<float>(std::string("Grid CrossEntropyLoss_output"), grad_input, 2, 2, n1);
}
#endif