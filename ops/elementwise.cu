#include "elementwise.cuh"

#include "math.h"

template <typename T> 
__global__ void MemoryCpyLinear(T* out, T* in, size_t max, size_t warpsize) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max; i += gridDim.x * blockDim.x)
        out[i] = in[i%warpsize];
    __syncthreads();
}

template <typename T>
__global__ void device_copy_pooler(T* out, 
                                   T* in, 
                                   size_t hidden_size, 
                                   size_t seq_length,
                                   size_t batchsize){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x ; 
           i < batchsize * hidden_size; i += gridDim.x * blockDim.x){
        size_t num_batch = i / hidden_size;
        out[i] = in[num_batch * seq_length * hidden_size + i%hidden_size];
    }
    __syncthreads();
}

template <typename T>
void copy_pooler(T* &output, T* tensor, global_handle* handle){
    output = handle->global_malloc_manage_float.get_new_head_point(
                            handle->batchsize * handle->hidden_size);

    dim3 threads(handle->hidden_size, 1, 1);
    dim3 blocks(min(long(65535), handle->batchsize), 1, 1);
    device_copy_pooler<<<blocks, threads, 0, handle->cal_stream>>>(
                                    output,
                                    tensor,
                                    handle->hidden_size,
                                    handle->seq_length,
                                    handle->batchsize);
}

template 
void copy_pooler<float>(float* &output, float* tensor, global_handle* handle);

template
__global__ void MemoryCpyLinear<float>(float* out, float* in, size_t max, size_t warpsize);

template <typename T> 
__device__ void BertTranspose (T* out, 
                              T *in, 
                              size_t batchsize, 
                              size_t seq_length, 
                              size_t num_heads, 
                              long total_length,
                              bool muti_head) {
  for(size_t index = blockIdx.x; index < total_length; index += gridDim.x){
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

template <typename T>
__global__ void FusionTranspose (T* out,
                                T* in, 
                                int num, 
                                size_t batchsize, 
                                size_t seq_length,
                                size_t num_heads,
                                size_t total_length,
                                size_t total, 
                                bool muti_head) {
    for(int i = 0; i < num; i++)
        BertTranspose(out + i*total, in + i*total, batchsize, seq_length, num_heads, total_length, muti_head);
    __syncthreads();
}

template <typename T>
void op_FusionTranspose::forward (T* &out,
                                T* in, 
                                int num, 
                                bool muti_head) {
    out = handle->global_malloc_manage_float.get_new_head_point(
            handle->batchsize * handle->seq_length * handle->hidden_size * 3);
    dim3 threads(handle->hidden_size, 1, 1);
    dim3 blocks(min(long(65535), handle->batchsize * handle->seq_length), 1, 1);

    FusionTranspose<<<blocks, threads, 0, handle->cal_stream>>>(
                        out,
                        in,
                        num, 
                        handle->batchsize, 
                        handle->seq_length,
                        handle->num_attention_heads,
                        handle->batchsize * handle->seq_length,
                        handle->batchsize * handle->seq_length * handle->hidden_size, 
                        muti_head);
}

template
void op_FusionTranspose::forward<float>(float* &out,
                                        float* in, 
                                        int num, 
                                        bool muti_head);

template <typename T>
__global__ void mask (
                    T* tensor, 
                    int* mask, 
                    float number, 
                    size_t max_num, 
                    size_t batchsize, 
                    size_t seq_length) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x){
        size_t index = seq_length * ( i / ( max_num / batchsize )) + i % seq_length;
        if(mask != nullptr)
            tensor[i] = tensor[i]/number + (1 - mask[index]) * -10000.0;
        else
            tensor[i] = tensor[i]/number;
    } 
    __syncthreads();
}

template <typename T>
void op_Mask_Add::forward (
                    T* tensor, 
                    int* attention_mask, 
                    float number) {
    size_t seq_length = handle->seq_length;
    size_t batchsize = handle->batchsize;
    size_t num_attention_heads = handle->num_attention_heads;

    dim3 threads(1024, 1, 1);
    dim3 blocks(min( (long)65535, 
            seq_length*seq_length*batchsize*num_attention_heads / 1024) + 1, 1, 1);
    mask<<<blocks, threads, 0, handle->cal_stream>>>(
                  tensor, 
                  attention_mask, 
                  number, 
                  batchsize * seq_length * num_attention_heads * seq_length, 
                  batchsize, 
                  seq_length); 
}

template
void op_Mask_Add::forward<float>(
                            float* tensor, 
                            int* mask, 
                            float number);

template <typename T>
__global__ void gelu (T* tensor,  size_t max_num) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] = tensor[i] * 0.5f * (1.0f + erff(tensor[i] / sqrtf(2.0)));
    __syncthreads();
}

template <typename T>
void op_Gelu::forward (T* tensor, size_t max_num) {
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long)65535, max_num / 1024) + 1, 1, 1);
    gelu<<<blocks, threads, 0, handle->cal_stream>>>(
                                     tensor, 
                                     max_num); 
}

template
void op_Gelu::forward<float>(float* tensor, size_t max_num);

template <typename T>
__global__ void Tanh (T* tensor,  size_t max_num) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < max_num; i += gridDim.x * blockDim.x)
        tensor[i] = tanh(tensor[i]);
    __syncthreads();
}

template <typename T>
void op_Tanh::forward (T* tensor, size_t max_num){
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long)65535, max_num / 1024) + 1, 1, 1);
    Tanh<<<blocks, threads, 0, handle->cal_stream>>>(
                                    tensor, 
                                    max_num);
}

template
void op_Tanh::forward<float>(float* tensor,  size_t max_num);
