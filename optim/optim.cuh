#ifndef BERT_CUDA_OPTIM
#define BERT_CUDA_OPTIM
#include "../utils/manager.cuh"

void apply_sgd_running_time(float* input,float* grad, size_t n, float learning_rate, global_handle *handl);
void apply_adam_running_time(float *input, float *grad, size_t n, float *m_t, float *v_t, float beta_1_t,
                             float beta_2_t, global_handle *handle, float learning_rate, float weight_decay_rate, float beta_1,
                             float beta_2, float epsilon, int step);
void apply_momentum_running_time(float *input, float *grad, size_t n, float *v, float lr, float beta, global_handle *handle, int step);

#endif