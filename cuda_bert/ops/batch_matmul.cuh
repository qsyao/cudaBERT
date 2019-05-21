#ifndef BERT_CUDA_BATCH_MATMUL
#define BERT_CUDA_BATCH_MATMUL

#include "op_kernel.cuh"
#include "../utils/manager.cuh"

class Query_Key : public op_kernel {
  public:
    Query_Key(global_handle* handle)
                : op_kernel(handle) {
        checkCudaErrors(cudaMalloc(
                        (void **) &query_array, 
                        sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));
        checkCudaErrors(cudaMalloc(
                        (void **) &key_array, 
                        sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));
        checkCudaErrors(cudaMalloc(
                        (void **) &out_array, 
                        sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));                  
    }

    void forward(const float* query,
                const float* key,
                float number,
                float* &output);

    void backward() {}

    ~Query_Key(){
      checkCudaErrors(cudaFree(query_array));
      checkCudaErrors(cudaFree(key_array));
      checkCudaErrors(cudaFree(out_array));
    }

  private:
    const float **query_array;
    const float **key_array;
    float **out_array;
};

class Prob_Value : public op_kernel {
public:
  Prob_Value(global_handle* handle)
              : op_kernel(handle) {
      checkCudaErrors(cudaMalloc(
                      (void **) &prob_array, 
                      sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));
      checkCudaErrors(cudaMalloc(
                      (void **) &value_array, 
                      sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));
      checkCudaErrors(cudaMalloc(
                      (void **) &out_array, 
                      sizeof(float *) * handle->max_batchsize * handle->num_attention_heads));                  
    }

  void forward(const float* prob,
              const float* value,
              float* &output);

  void backward() {}

  ~Prob_Value(){
    checkCudaErrors(cudaFree(prob_array));
    checkCudaErrors(cudaFree(value_array));
    checkCudaErrors(cudaFree(out_array));
  }


private:
  const float **prob_array;
  const float **value_array;
  float **out_array;
};

#endif