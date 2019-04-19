#include "embedding.cuh"

template <typename T> __global__
void DeviceApplyEmbeddings (const int* words, 
                           int* token_type, 
                           int* positions,
                           int total_length,
                           T* output, 
                           T* embedding_words, 
                           T* embedding_token_types, 
                           T* embedding_positions) {
    int warp_size = blockDim.x;
    for(int index = blockIdx.x; index < total_length; index += gridDim.x){
        int curnIdx = index * warp_size + threadIdx.x;
        output[curnIdx] = embedding_words[warp_size * words[index] + threadIdx.x] +
                          embedding_token_types[warp_size * token_type[index] + threadIdx.x] +
                          embedding_positions[warp_size * positions[index] + threadIdx.x];     
    }
    __syncthreads();
}

template <typename T>
void Embedding::forward (T* &output,
                         int *words, 
                         int *token_type, 
                         int *position) {
    // MemcpyHostToDevice for inputs
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;

    int total_length = batchsize * seq_length;

    output = handle->global_malloc_manage_float.get_new_head_point(
                                         batchsize *
                                         seq_length *
                                         hidden_size);

    dim3 threads(hidden_size, 1, 1);
    dim3 blocks(min(65536, total_length), 1, 1);
    DeviceApplyEmbeddings<<<blocks, threads, 0, handle->cal_stream>>>(
                                               words,
                                               token_type, 
                                               position,
                                               total_length,
                                               output, 
                                               word_embedding, 
                                               position_embedding, 
                                               token_type_embedding);
    //debug_tensor_gpu<float>(std::string("after embedding add : "), output, 10, handle->hidden_size, 5);

    layernorm->forward( output, 
                        output,
                        total_length, 
                        hidden_size);
    
    //debug_tensor_gpu<float>(std::string("look_up_embedding on GPU"), output, 10, 768, 11*batchsize);
}

template
void Embedding::forward<float>(
                         float* &output,
                         int *words, 
                         int *token_type, 
                         int *position);
