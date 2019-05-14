#ifndef BERT_CUDA_OPTIM
#define BERT_CUDA_OPTIM
#include "../utils/manager.cuh"

void apply_sgd_running_time(float* input,float* grad, size_t n, float learning_rate, global_handle *handl);
void apply_adam_running_time(float *input, float *grad, size_t n, float *m_t, float *v_t, float beta_1_t,
                             float beta_2_t, global_handle *handle, float learning_rate = 0.001, float weight_decay_rate = 0.0, float beta_1 = 0.9,
                             float beta_2 = 0.999, float epsilon = 1e-8, int step = 0);
#endif