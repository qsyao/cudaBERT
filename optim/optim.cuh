#ifndef BERT_CUDA_OPTIM
#define BERT_CUDA_OPTIM
#include "../utils/manager.cuh"

void apply_sgd_running_time(float* input,float* grad, size_t n, global_handle *handl);

#endif