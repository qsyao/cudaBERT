#include "elementwise.cuh"

template <typename T> 
__global__ void MemoryCpyLinear(T* out, T* in, int max, int warpsize) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max; i += gridDim.x * blockDim.x)
        out[i] = in[i%warpsize];
    __syncthreads();
}

template
__global__ void MemoryCpyLinear<float>(float* out, float* in, int max, int warpsize);

template <typename T> 
__device__ void BertTranspose (T* out, 
                              T *in, 
                              size_t batchsize, 
                              size_t seq_length, 
                              size_t num_heads, 
                              long total_length,
                              bool muti_head) {
  for(int index = blockIdx.x; index < total_length; index += gridDim.x){
    size_t remain = blockDim.x / num_heads;
    size_t size_head = seq_length * remain;
    size_t num_batch = index / seq_length;
    size_t size_batch = blockDim.x * seq_length;
    size_t num_head = threadIdx.x / remain;
    size_t num_seq = index % seq_length;
    size_t size_seq = remain;
    size_t forward_index = num_batch*size_batch + num_head*size_head + num_seq*size_seq + threadIdx.x%remain;
    size_t backward_index = blockDim.x * index + threadIdx.x;
    if(muti_head){    
        out[forward_index] = in[backward_index];
    }
    else{
        out[backward_index] = in[forward_index];
    }
  }
  __syncthreads();
}

template
__device__ void BertTranspose<float>(float* out, 
                              float *in, 
                              size_t batchsize, 
                              size_t seq_length, 
                              size_t num_heads, 
                              long total_length,
                              bool muti_head);

template <typename T>
__global__ void FusionTranspose (T* out,
                                T* in, 
                                int num, 
                                size_t batchsize, 
                                size_t seq_length,
                                size_t num_heads,
                                long total_length,
                                long total, 
                                bool muti_head) {
    for(int i = 0; i < num; i++)
        BertTranspose(out + i*total, in + i*total, batchsize, seq_length, num_heads, total_length, muti_head);
    __syncthreads();
}

template
__global__ void FusionTranspose<float>(float* out,
                                float* in, 
                                int num, 
                                size_t batchsize, 
                                size_t seq_length,
                                size_t num_heads,
                                long total_length,
                                long total, 
                                bool muti_head);

template <typename T>
__global__ void BertDiv (T* tensor, float number, int max_num) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] /= number;
    __syncthreads();
}

template
__global__ void BertDiv<float>(float* tensor, float number, int max_num);

template <typename T>
__global__ void BertAdd (T* tensor, T* add, int max_num) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] += add[i];
    __syncthreads();
}

template
__global__ void BertAdd<float>(float* tensor, float* add, int max_num);

template <typename T>
__global__ void Attention_Mask_Add_Merge_div_only_Add (
                                T* tensor, 
                                int* mask, 
                                float number, 
                                int max_num, 
                                int batchsize, 
                                int seq_length) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x){
        int index = seq_length * ( i / ( max_num / batchsize )) + i % seq_length;
        tensor[i] = tensor[i]/number + (1 - mask[index]) * -10000.0;
    } 
    __syncthreads();
}

template
__global__ void Attention_Mask_Add_Merge_div_only_Add<float>(
                                float* tensor, 
                                int* mask, 
                                float number, 
                                int max_num, 
                                int batchsize, 
                                int seq_length);

template <typename T>
__global__ void Attention_Mask_Add_Merge_div_only_div (
                                  T* tensor, 
                                  int* mask, 
                                  float number, 
                                  int max_num, 
                                  int batchsize, 
                                  int seq_length) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x){
        tensor[i] = tensor[i] / number;
    } 
    __syncthreads();
}

template
__global__ void Attention_Mask_Add_Merge_div_only_div<float>(
                                  float* tensor, 
                                  int* mask, 
                                  float number, 
                                  int max_num, 
                                  int batchsize, 
                                  int seq_length);

template <typename T>
__global__ void BertGelu (T* tensor,  int max_num) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] = tensor[i] * 0.5f * (1.0f + erff(tensor[i] / sqrtf(2.0)));
    __syncthreads();
}

template
__global__ void BertGelu<float>(float* tensor,  int max_num);

template <typename T>
__global__ void BertTanh (T* tensor,  int max_num) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] = tanh(tensor[i]);
    __syncthreads();
}

template
__global__ void BertTanh<float>(float* tensor,  int max_num);
