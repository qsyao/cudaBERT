#ifndef LAYERNORM_BERT_CUDA
#define LAYERNORM_BERT_CUDA

#include "op_kernel.cuh"
#include "../utils/common.h"
#include "../optim/optim.cuh"

class op_LayerNorm : public op_kernel{
  public:
    op_LayerNorm(std::string key_gamma, 
                 std::string key_beta,
                 global_handle* handle,
                 int n = -1)
                 : op_kernel(handle) {
        std::vector<std::string> keys = {key_gamma};
        tagged_tensor* tt = look_up_tts(handle->tts, keys);
        gamma = tt->gpu_mem;
        keys = {key_beta};
        tt = look_up_tts(handle->tts, keys);
        beta = tt->gpu_mem;
        if(handle->is_train) {
            if(handle->optim_method == "sgd") {
                learning_rate = handle->learning_rate;
            }
            else if(handle->optim_method == "adam") {
                learning_rate = handle->learning_rate;
                weight_decay_rate = handle->weight_decay_rate;
                beta_1 = handle->beta_1;
                beta_2 = handle->beta_2;
                gamma_m_t = handle->global_malloc_manage_float.get_new_head_point(n);
                gamma_v_t = handle->global_malloc_manage_float.get_new_head_point(n);

                beta_m_t = handle->global_malloc_manage_float.get_new_head_point(n);
                beta_v_t = handle->global_malloc_manage_float.get_new_head_point(n);
                adam_epsilon = handle->epsilon;
                step = 0;
            }
        }
    }

    ~op_LayerNorm();

    template<typename T> 
    void forward(
    /* 
        Fuse op_Add in LayerNorm For BERT Only
    */
                T* &output,
                T* input,
                size_t n1,
                size_t n2,
                T* merge_add = nullptr);

    template<typename T>
    void backward(T *dout, size_t n1, size_t n2);

    void update_weights(size_t n);

  public:
    float learning_rate;
    size_t warpsize;
    double epsilon = 1e-12;
    float* gamma;
    float* beta;
    float* grad_input;
    float* grad_gamma;
    float* grad_beta;
    float* mean;
    float* invvar;

    float beta_1;
    float beta_2;
    float weight_decay_rate;
    float adam_epsilon;
    float beta_1_t;
    float beta_2_t;
    float *gamma_m_t;
    float *gamma_v_t;
    float *beta_m_t;
    float *beta_v_t;
    int step = 0;
};

#endif
