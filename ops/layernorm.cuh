#ifndef LAYERNORM_BERT_CUDA
#define LAYERNORM_BERT_CUDA

#include "op_kernel.cuh"
#include "../utils/common.h"
#include "../optim/optim.cuh"

class op_LayerNorm : public op_kernel{
  public:
    op_LayerNorm(std::string key_gamma, 
                 std::string key_beta,
                 global_handle* handle)
                 : op_kernel(handle) {
        std::vector<std::string> keys = {key_gamma};
        tagged_tensor* tt = look_up_tts(handle->tts, keys);
        gamma = tt->gpu_mem;
        keys = {key_beta};
        tt = look_up_tts(handle->tts, keys);
        beta = tt->gpu_mem;
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
    size_t warpsize;
    double epsilon = 1e-12;
    float* gamma;
    float* beta;
    float* grad_input;
    float* grad_gamma;
    float* grad_beta;
    float* mean;
    float* invvar;
};

#endif
