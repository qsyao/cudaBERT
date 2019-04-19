#ifndef LINEAR_CUDA_BERT
#define LINEAR_CUDA_BERT

#include "elementwise.cuh"
#include "op_kernel.cuh"
#include "../utils/common.h"

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
    
    void backward();

    void update();

  private:
    size_t n, k; // Shape of Weight: [n, k]

    float* kernel;
    float* bias;
};

class op_BatchedLinear : public op_kernel{
  public:
    op_BatchedLinear( std::string key_query_kernel, 
                      std::string key_query_bias,
                      std::string key_key_kernel, 
                      std::string key_key_bias,
                      std::string key_val_kernel, 
                      std::string key_val_bias,
                      global_handle* handle)
                        : op_kernel(handle)  {
        std::vector<tagged_tensor *> tts = handle->tts;

        std::vector<std::string> keys = {key_query_kernel};
        tagged_tensor* tt = look_up_tts(tts, keys);
        query_kernel = tt->gpu_mem;
        keys = {key_query_bias};
        tt = look_up_tts(tts, keys);
        query_bias = tt->gpu_mem;

        keys = {key_key_kernel};
        tt = look_up_tts(tts, keys);
        key_kernel = tt->gpu_mem;
        keys = {key_key_bias};
        tt = look_up_tts(tts, keys);
        key_bias = tt->gpu_mem;

        keys = {key_val_kernel};
        tt = look_up_tts(tts, keys);
        val_kernel = tt->gpu_mem;
        keys = {key_val_bias};
        tt = look_up_tts(tts, keys);
        val_bias = tt->gpu_mem;

        size_t hidden_size = handle->hidden_size;

        float *key, *value;
        checkCudaErrors(cudaMalloc((void**)&batch_attentin_weights, 
                sizeof(float) * hidden_size * hidden_size * 3));
        key = batch_attentin_weights + 1 * hidden_size * hidden_size;
        value = batch_attentin_weights + 2 * hidden_size * hidden_size;

        dim3 threads(512, 1, 1);
        dim3 blocks(hidden_size * hidden_size/512 + 1, 1, 1);
        MemoryCpyLinear<<<blocks, threads>>>(batch_attentin_weights, 
                                              query_kernel,
                                              hidden_size * hidden_size,
                                              hidden_size * hidden_size);
        MemoryCpyLinear<<<blocks, threads>>>(key, 
                                              key_kernel,
                                              hidden_size * hidden_size,
                                              hidden_size * hidden_size);
        MemoryCpyLinear<<<blocks, threads>>>(value, 
                                              val_kernel,
                                              hidden_size * hidden_size,
                                              hidden_size * hidden_size);

        checkCudaErrors(cudaFree(query_kernel));
        checkCudaErrors(cudaFree(key_kernel));
        checkCudaErrors(cudaFree(val_kernel));
        query_kernel = batch_attentin_weights;
        key_kernel = key;
        val_kernel = value;
    }

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

    void backward();

    void update();

  private:
    size_t n, k; // Shape of Weight: [n, k]

    float* query_kernel;
    float* query_bias;
    float* key_kernel; 
    float* key_bias;
    float* val_kernel; 
    float* val_bias;

    float* batch_attentin_weights;
};

#endif
