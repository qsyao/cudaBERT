#include "optim.cuh"
#include "../utils/common.h"

__global__ void cu_apply_sgd_running_time(float *input, float *grad, size_t n, float lr) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        input[i] -= lr * grad[i];
    __syncthreads();
}

void apply_sgd_running_time(float *input, float *grad, size_t n, float learning_rate, global_handle *handle) {
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long) 65535, n / 1024 + 1), 1, 1);
    cu_apply_sgd_running_time << < blocks, threads, 0, handle->cal_stream >> > (input, grad, n, learning_rate);
}

__global__ void cu_apply_momentum_running_time(float *input, float *grad, size_t n, float *v, float lr, float beta, int step) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if(step == 0)
            v[i] = grad[i];
        else
            v[i] = beta * v[i] + grad[i];
        input[i] -= lr * v[i];
//        if(step == 0)
//            v[i] = lr * grad[i];
//        else
//            v[i] = beta * v[i] + lr * grad[i];
//        input[i] -= v[i];
    }
    __syncthreads();
}

void apply_momentum_running_time(float *input, float *grad, size_t n, float *v, float learning_rate, float beta, global_handle *handle, int step) {

    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long) 65535, n / 1024 + 1), 1, 1);
    cu_apply_momentum_running_time << < blocks, threads, 0, handle->cal_stream >> > (input, grad, n, v, learning_rate, beta, step);
}

__global__ void cu_apply_sgd_running_time(float *input, float *grad, size_t n, float *m_t, float *v_t, float beta_1_t,
                                          float beta_2_t, float learning_rate_t, float weight_decay_rate,
                                          float beta_1,
                                          float beta_2, float epsilon, int step) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if(step == 0) {
            m_t[i] = (1.0 - beta_1) * grad[i];
            v_t[i] = (1.0 - beta_2) * grad[i] * grad[i];
        }
        else {
            m_t[i] = beta_1 * m_t[i] + (1.0 - beta_1) * grad[i];
            v_t[i] = beta_2 * v_t[i] + (1.0 - beta_2) * grad[i] * grad[i];
        }
        input[i] -= learning_rate_t * ((m_t[i] / (sqrtf(v_t[i]) + epsilon)) + weight_decay_rate * input[i]);
    }
    __syncthreads();
}

//TODO: 验证weight_decay_rate
void apply_adam_running_time(float *input, float *grad, size_t n, float *m_t, float *v_t, float beta_1_t,
                             float beta_2_t, global_handle *handle, float learning_rate, float weight_decay_rate, float beta_1,
                             float beta_2, float epsilon, int step) {
    beta_1_t = beta_1_t * beta_1;
    beta_2_t = beta_2_t * beta_2;

    float learning_rate_t = learning_rate * sqrt(1.0 - beta_2_t) / (1.0 - beta_1_t);

//    std::cout << "beta_1_t: " << beta_1_t << std::endl;
//    std::cout << "beta_2_t: " << beta_2_t << std::endl;
//    std::cout << "step: " << step << std::endl;
//    std::cout << "learning_rate_t: " << learning_rate_t << std::endl;

    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long) 65535, n / 1024 + 1), 1, 1);
    cu_apply_sgd_running_time << < blocks, threads, 0, handle->cal_stream >> >
                                                       (input, grad, n, m_t, v_t, beta_1_t, beta_2_t, learning_rate_t, weight_decay_rate, beta_1, beta_2, epsilon, step);

}
