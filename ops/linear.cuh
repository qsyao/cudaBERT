#ifndef LINEAR_CUDA_BERT
#define LINEAR_CUDA_BERT

#include "op_kernel.cuh"
#include "../utils/common.h"

template <typename T>
__global__ void MemoryCpyLinear(T* out, T* in, size_t max, size_t warpsize, float mul = 1.0) ;

class op_Linear : public op_kernel{
  public:
    op_Linear(std::string key_kernel, 
              std::string key_bias,
              global_handle* handle)
                 : op_kernel(handle) {
        std::vector<std::string> keys = {key_kernel};
        tagged_tensor* tt = look_up_tts(handle->tts, keys);
        kernel = tt->gpu_mem;
        keys = {key_bias};
        tt = look_up_tts(handle->tts, keys);
        bias = tt->gpu_mem;
    }

    ~op_Linear();

    template <typename T>
    void forward(
                T* &output, 
                T* input, 
                size_t n, 
                size_t k, 
                size_t m,
                bool is_prepare=false,
                bool debug=false);

    template<typename T>
    void backward(T *dout, size_t n,
                  size_t k,
                  size_t m);

    void update();

public:
    size_t n, k; // Shape of Weight: [n, k]

    float *kernel;
    float *bias;
public:
    float *grad_input;
    float *grad_kernel;
    float *grad_bias;
};

class op_BatchedLinear : public op_kernel{
  public:
    op_BatchedLinear( std::string key_query_kernel, 
                      std::string key_query_bias,
                      std::string key_key_kernel, 
                      std::string key_key_bias,
                      std::string key_val_kernel, 
                      std::string key_val_bias,
                      global_handle* handle);

    ~op_BatchedLinear();

    template <typename T>
    void forward(
                T* &output, 
                T* input, 
                size_t n, 
                size_t k, 
                size_t m,
                bool is_prepare=false,
                bool debug=false);

    template<typename T>
    void backward(T *dout, T *grad_short_cut, size_t n, size_t k, size_t m);

    void update();

public:
    size_t n, k; // Shape of Weight: [n, k]

    float* query_kernel;
    float* query_bias;
    float* key_kernel; 
    float* key_bias;
    float* val_kernel; 
    float* val_bias;

    float* batch_attentin_weights;
public:
    float* grad_query_input;
    float* grad_query_kernel;
    float* grad_query_bias;

    float* grad_key_input;
    float* grad_key_kernel;
    float* grad_key_bias;

    float* grad_val_input;
    float* grad_val_kernel;
    float* grad_val_bias;
    float* grad_input;
};

#endif
