#ifndef LINEAR_CUDA_BERT
#define LINEAR_CUDA_BERT

#include "op_kernel.cuh"
#include "../utils/common.h"
#include "../optim/optim.cuh"

template <typename T>
__global__ void MemoryCpyLinear(T* out, T* in, size_t max, size_t warpsize, float mul = 1.0) ;

class op_Linear : public op_kernel{
  public:
    op_Linear(std::string key_kernel, 
              std::string key_bias,
              global_handle* handle,
              int n1 = -1,
              int n2 = -1)
                 : op_kernel(handle) {
        std::vector<std::string> keys = {key_kernel};
        tagged_tensor* tt = look_up_tts(handle->tts, keys);
        kernel = tt->gpu_mem;
        keys = {key_bias};
        tt = look_up_tts(handle->tts, keys);
        bias = tt->gpu_mem;
        if(handle->is_train) {
            if(handle->optim_method == "sgd") {
                learning_rate = handle->learning_rate;
            }
            else if(handle->optim_method == "adam") {
                learning_rate = handle->learning_rate;
                weight_decay_rate = handle->weight_decay_rate;
                beta_1 = handle->beta_1;
                beta_2 = handle->beta_2;

                kernel_m_t = handle->global_malloc_manage_float.get_new_head_point(n1);
                kernel_v_t = handle->global_malloc_manage_float.get_new_head_point(n1);

                bias_m_t = handle->global_malloc_manage_float.get_new_head_point(n2);
                bias_v_t = handle->global_malloc_manage_float.get_new_head_point(n2);
                epsilon = handle->epsilon;
                step = 0;
            }
        }
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

    void update_weights(size_t n1, size_t n2);

public:
    size_t n, k; // Shape of Weight: [n, k]

    float *kernel;
    float *bias;
    float beta_1;
    float beta_2;
    float weight_decay_rate;
    float epsilon;
    float beta_1_t;
    float beta_2_t;
    float *kernel_m_t;
    float *kernel_v_t;
    float *bias_m_t;
    float *bias_v_t;
    int step = 0;
public:
    float learning_rate;
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
                      global_handle* handle,
                      int n1 = -1,
                      int n2 = -1);

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

    void update_weights(size_t n1, size_t n2);

public:
    size_t n, k; // Shape of Weight: [n, k]

    float* query_kernel;
    float* query_bias;
    float* key_kernel; 
    float* key_bias;
    float* val_kernel; 
    float* val_bias;
    float beta_1;
    float beta_2;
    float weight_decay_rate;
    float epsilon;
    float beta_1_t;
    float beta_2_t;
    float *query_kernel_m_t;
    float *query_kernel_v_t;
    float *query_bias_m_t;
    float *query_bias_v_t;
    float* key_kernel_m_t;
    float* key_kernel_v_t;
    float* key_bias_m_t;
    float* key_bias_v_t;
    float* val_kernel_m_t;
    float* val_kernel_v_t;
    float* val_bias_m_t;
    float* val_bias_v_t;
    int step = 0;

    float* batch_attentin_weights;
public:
    float learning_rate;
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
