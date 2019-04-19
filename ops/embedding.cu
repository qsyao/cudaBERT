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
void HostApplyEmbeddings (global_manager *handle, 
                         T* &output,
                         int *words, 
                         int *token_type, 
                         int* &attention_mask) {
    // MemcpyHostToDevice for inputs

    std::vector<tagged_tensor *> tts = handle->tts;
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;

    int total_length = batchsize * seq_length;
    int *word_gpu, *token_type_gpu, *positions_gpu, *mask_gpu;

    int positions[total_length];
    for( int i = 0; i < total_length; i++){
        positions[i] = i % seq_length;
    }
    
    int* host_input_package;
    checkCudaErrors(cudaMallocHost((void **)&host_input_package, 4*total_length*sizeof(int)));
    memcpy(host_input_package, words, total_length*sizeof(int));
    memcpy(host_input_package + total_length, token_type, total_length*sizeof(int));
    memcpy(host_input_package + 2*total_length, positions, total_length*sizeof(int));

    word_gpu = handle->global_malloc_manage_int.get_new_head_point(total_length);
    token_type_gpu = handle->global_malloc_manage_int.get_new_head_point(total_length);
    positions_gpu = handle->global_malloc_manage_int.get_new_head_point(total_length);
    if(attention_mask != nullptr){
        mask_gpu = handle->global_malloc_manage_int.get_new_head_point(total_length);
        memcpy(host_input_package + 3*total_length, attention_mask, total_length*sizeof(int));
        checkCudaErrors(cudaMemcpyAsync(word_gpu, host_input_package, 4*total_length*sizeof(int), cudaMemcpyHostToDevice));
        attention_mask = mask_gpu;
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(word_gpu, host_input_package, 3*total_length*sizeof(int), cudaMemcpyHostToDevice));
    }
    cudaFreeHost(host_input_package);
    //debug_tensor_gpu<int>(std::string("memcpy: "), word_gpu, 10, seq_length, batchsize*3);
    
    // HostApplyEmbeddings
    T *embedding_words, *embedding_positions, *embedding_token_types;
    std::vector<std::string> keys = {"embeddings", "word"};
    embedding_words = look_up_tts(tts, keys)->gpu_mem;
    keys = {"embeddings", "position"};
    embedding_positions = look_up_tts(tts, keys)->gpu_mem;
    keys = {"embeddings", "token_type"};
    embedding_token_types = look_up_tts(tts, keys)->gpu_mem;     

    dim3 threads(hidden_size, 1, 1);
    dim3 blocks(min(65536, total_length), 1, 1);
    DeviceApplyEmbeddings<<<blocks, threads, 0, handle->get_cal_stream()>>>(
                                               word_gpu,
                                               token_type_gpu, 
                                               positions_gpu,
                                               total_length,
                                               output, 
                                               embedding_words, 
                                               embedding_token_types, 
                                               embedding_positions);
    //debug_tensor_gpu<float>(std::string("after embedding add : "), output, 10, handle->hidden_size, 5);

    keys = {"embeddings_LayerNorm_gamma"};
    T* gamma = look_up_tts(tts, keys)->gpu_mem;
    keys = {"embeddings_LayerNorm_beta"};
    T* beta = look_up_tts(tts, keys)->gpu_mem;
    HostApplyLayerNorm<T, T>(handle,
                             output, 
                             output,
                             total_length, 
                             hidden_size, 
                             1e-12, 
                             gamma, 
                             beta);
    
    //debug_tensor_gpu<float>(std::string("look_up_embedding on GPU"), output, 10, 768, 11*batchsize);
}

template
void HostApplyEmbeddings<float>(global_manager *handle, 
                         float* &output,
                         int *words, 
                         int *token_type, 
                         int* &attention_mask);
