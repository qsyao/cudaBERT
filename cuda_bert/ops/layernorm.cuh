#ifndef LAYERNORM_BERT_CUDA
#define LAYERNORM_BERT_CUDA

#include "op_kernel.cuh"
#include "../utils/common.h"

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

    ~op_LayerNorm(){
      checkCudaErrors(cudaFree(gamma));
      checkCudaErrors(cudaFree(beta));
    }

    template<typename T> 
    void forward(
    /* 
        Fuse op_Add in LayerNorm For BERT Only
    */
                T* output,
                T* input,
                size_t n1,
                size_t n2,
                T* merge_add = nullptr); 

    void backward();

    void update_weights();

  private:
    size_t warpsize;
    double epsilon = 1e-12;
    float* gamma;
    float* beta;
};

#endif
