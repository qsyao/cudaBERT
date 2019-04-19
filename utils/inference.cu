#include "inference.cuh"

template <typename T> 
void BERT_Attention (global_manager *handle, 
                    T* &tensor,
                    size_t num_layer, 
                    int* attention_mask) {
    dict_weights weights = handle->weights;
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;                    
    size_t hidden_size = handle->hidden_size;
    size_t num_attention_heads = handle->num_attention_heads;
    
    T *query;

    query = handle->global_malloc_manage_float.get_new_head_point(3*batchsize*seq_length*hidden_size);
    //key = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*hidden_size);
    //val = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*hidden_size);
/*
    Linear<T>(handle, query, tensor, weights["attention_self_query_kernel"][num_layer],
          weights["attention_self_query_bias"][num_layer], batchsize*seq_length, hidden_size, hidden_size);
    Linear<T>(handle, key, tensor, weights["attention_self_key_kernel"][num_layer],
          weights["attention_self_key_bias"][num_layer], batchsize*seq_length, hidden_size, hidden_size);
    Linear<T>(handle, val, tensor, weights["attention_self_value_kernel"][num_layer],
          weights["attention_self_value_bias"][num_layer], batchsize*seq_length, hidden_size, hidden_size);
    //debug_tensor_gpu<T>(std::string("query"), query, 10, hidden_size, 11*batchsize);
*/
    Batch_Linear<T>(handle, 
          query, 
          tensor, 
          weights["attention_self_query_kernel"][num_layer],
          weights["attention_self_query_bias"][num_layer],
          weights["attention_self_key_kernel"][num_layer],
          weights["attention_self_key_bias"][num_layer],
          weights["attention_self_value_kernel"][num_layer],
          weights["attention_self_value_bias"][num_layer],
          batchsize * seq_length, 
          hidden_size,
          hidden_size);
    //debug_tensor_gpu<T>(std::string("query"), query, 10, hidden_size, batchsize);

    T *head_query, *head_key, *head_val;
    head_query = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*hidden_size);
    head_key = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*hidden_size);
    head_val = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*hidden_size);

    dim3 threads(hidden_size, 1, 1);
    dim3 blocks(min(long(65535), batchsize*seq_length), 1, 1);

    FusionTranspose<<<blocks, threads, 0, handle->get_cal_stream()>>>(
                                         head_query, 
                                         query, 
                                         3, 
                                         batchsize, 
                                         seq_length,
                                         num_attention_heads,
                                         batchsize * seq_length,
                                         batchsize * seq_length * hidden_size, 
                                         true);

    T *probs;
    probs = handle->global_malloc_manage_float.get_new_head_point(batchsize*num_attention_heads*seq_length*seq_length);
    std::vector<size_t> a_shape = {batchsize*num_attention_heads, seq_length, hidden_size/num_attention_heads};
    std::vector<size_t> b_shape = {batchsize*num_attention_heads, seq_length, hidden_size/num_attention_heads};
    std::vector<size_t> c_shape = {batchsize*num_attention_heads, seq_length, seq_length};
    matmul(handle->handle, head_query, a_shape, head_key, b_shape, probs, c_shape, false, true);
    //debug_tensor_gpu<T>(std::string("matmul key and query"), probs, 10, seq_length, 12*2*batchsize);

    dim3 div_threads(1024, 1, 1);
    dim3 div_blocks(min((long)65535, seq_length*seq_length*batchsize*num_attention_heads / 1024) + 1, 1, 1);
    //debug_tensor_gpu<int>(std::string("attention_mask"), attention_mask, 11, 11, batchsize);
    if (attention_mask == nullptr) {
        Attention_Mask_Add_Merge_div_only_div<<<div_blocks, div_threads, 0, handle->get_cal_stream()>>>(
                  probs, 
                  attention_mask, 
                  8.0, 
                  batchsize * seq_length * num_attention_heads * seq_length, 
                  batchsize, 
                  seq_length);
    }
    else{
        Attention_Mask_Add_Merge_div_only_Add<<<div_blocks, div_threads, 0, handle->get_cal_stream()>>>(
                  probs, 
                  attention_mask, 
                  8.0, 
                  batchsize * seq_length * num_attention_heads * seq_length, 
                  batchsize, 
                  seq_length);
    }
    
    //debug_tensor_gpu<T>(std::string("After mask_add"), probs, 11, 11*11, 12 * batchsize);

    HostApplySoftmax(handle, probs, batchsize*num_attention_heads*seq_length, seq_length);
    //debug_tensor_gpu<T>(std::string("Softmax"), probs, 10, seq_length*seq_length, batchsize);

    T *attention;
    attention = handle->global_malloc_manage_float.get_new_head_point(batchsize*hidden_size*seq_length);
    a_shape = {batchsize*num_attention_heads, seq_length, seq_length};
    b_shape = {batchsize*num_attention_heads, seq_length, hidden_size/num_attention_heads};
    c_shape = {batchsize*num_attention_heads, seq_length, hidden_size/num_attention_heads};
    matmul(handle->handle, probs, a_shape, head_val, b_shape, attention, c_shape);

    T *hidden_states, *temp;
    hidden_states = handle->global_malloc_manage_float.get_new_head_point(batchsize*hidden_size*seq_length);
    temp = handle->global_malloc_manage_float.get_new_head_point(batchsize*hidden_size*seq_length);
    FusionTranspose<<<blocks, threads, 0, handle->get_cal_stream()>>>(
                                         hidden_states, 
                                         attention, 
                                         1, 
                                         batchsize, 
                                         seq_length,
                                         num_attention_heads,
                                         batchsize*seq_length,
                                         batchsize * hidden_size * seq_length, 
                                         false);

    Linear<T>(handle, 
              temp, 
              hidden_states, 
              weights["attention_output_dense_kernel"][num_layer],
              weights["attention_output_dense_bias"][num_layer], 
              batchsize*seq_length, 
              hidden_size, 
              hidden_size,
              false);
    hidden_states = temp;

    T* gamma = weights["attention_output_LayerNorm_gamma"][num_layer];
    T* beta = weights["attention_output_LayerNorm_beta"][num_layer];
    HostApplyLayerNorm<T, T>(handle,
                            tensor, 
                            tensor, 
                            batchsize*seq_length, 
                            hidden_size, 
                            1e-12, 
                            gamma, 
                            beta, 
                            hidden_states);
}

template <typename T>
void BERT_Intermediate (global_manager *handle, 
                        T* tensor, 
                        T* intermediate,
                        size_t num_layer) {

    dict_weights weights = handle->weights;
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;
    size_t intermediate_size = handle->intermediate_size;

    Linear<T>(handle, 
              intermediate, 
              tensor, 
              weights["intermediate_dense_kernel"][num_layer],
              weights["intermediate_dense_bias"][num_layer], 
              batchsize * seq_length, 
              hidden_size, 
              intermediate_size);
            
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long)65535, intermediate_size*seq_length*batchsize / 1024) + 1, 1, 1);
    BertGelu<<<blocks, threads, 0, handle->get_cal_stream()>>>(
                                     intermediate, 
                                     intermediate_size * seq_length * batchsize);

}

template <typename T>
void BERT_Output (global_manager *handle, 
                 T* tensor, 
                 T* intermediate,
                 size_t num_layer) {
    
    dict_weights weights = handle->weights;
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;
    size_t intermediate_size = handle->intermediate_size; 
    //dump_tensor<float>(std::string("transpose_output_"+std::to_string(num_layer)), hidden_states, batchsize, seq_length, handle->hidden_size);
       
    T* temp;
    temp = handle->global_malloc_manage_float.get_new_head_point(batchsize*seq_length*intermediate_size);
    Linear<T>(handle, 
              temp, 
              intermediate, 
              weights["output_dense_kernel"][num_layer],
              weights["output_dense_bias"][num_layer], 
              batchsize * seq_length, 
              intermediate_size, 
              hidden_size);

    //dim3 threads(1024, 1, 1);
    //dim3 blocks(seq_length*batchsize*hidden_size / 1024 + 1, 1, 1);
    //BertAdd<<<blocks, threads>>>(tensor, temp, batchsize*seq_length*hidden_size);

    T* gamma = weights["output_LayerNorm_gamma"][num_layer];
    T* beta = weights["output_LayerNorm_beta"][num_layer];
    HostApplyLayerNorm<T, T>(handle,
                            tensor, 
                            tensor, 
                            batchsize*seq_length, 
                            hidden_size, 
                            1e-12, 
                            gamma, 
                            beta, 
                            temp);
    
}

template <typename T> 
void BERT_Layer (global_manager *handle, 
                T* &tensor,
                size_t num_layer,
                int* attention_mask) {
    handle->global_malloc_manage_float.record_layer_start();

    BERT_Attention<T>(handle, tensor, num_layer, attention_mask);
    //ebug_tensor_gpu<T>(std::string("bert_attention_output"), tensor, 10, handle->hidden_size, handle->batchsize);

    T* intermediate;
    intermediate = handle->global_malloc_manage_float.get_new_head_point(
                                                        handle->batchsize *
                                                        handle->seq_length *
                                                        handle->intermediate_size);

    BERT_Intermediate<T>(handle, tensor, intermediate, num_layer);
    //debug_tensor_gpu<T>(std::string("bert_intermediate_output"), intermediate, 10, handle->intermediate_size, handle->batchsize);

    BERT_Output<T>(handle, tensor, intermediate, num_layer);
    
    cudaStreamSynchronize(handle->get_cal_stream());
    handle->global_malloc_manage_float.reuse_layer_mem();
}

template <typename T> 
void BERT_Pooler (global_manager *handle, 
                 T* &tensor, 
                 T* &pooled_output) {
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length; 
    size_t hidden_size = handle->hidden_size;

    T* first_token;
    first_token = handle->global_malloc_manage_float.get_new_head_point(batchsize*hidden_size);
    for (int i = 0; i < batchsize; i++) {
        checkCudaErrors(cudaMemcpyAsync(first_token + i*hidden_size, 
                                   tensor + i*hidden_size*seq_length, 
                                   hidden_size*sizeof(T), 
                                   cudaMemcpyDeviceToDevice,
                                   handle->get_cal_stream()));
    }

    std::vector<std::string> keys = {"pooler_dense_kernel"};
    T* kernel = look_up_tts(handle->tts, keys)->gpu_mem;
    keys = {"pooler_dense_bias"};
    T* bias = look_up_tts(handle->tts, keys)->gpu_mem;
    Linear<T>(handle, 
              pooled_output, 
              first_token, 
              kernel, 
              bias, 
              batchsize, 
              hidden_size, 
              hidden_size, 
              false);

    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long)65535, hidden_size*batchsize / 1024) + 1, 1, 1);
    BertTanh<<<blocks, threads, 0, handle->get_cal_stream()>>>(
                                    pooled_output, 
                                    hidden_size*batchsize);
}

template <typename T>
void BERT_Inference (global_manager *handle, 
                    T* &tensor,  
                    T* &pooled_output,
                    int* words, 
                    int* token_types, 
                    size_t batchsize, 
                    size_t seq_length, 
                    int* attention_mask) {
    handle->set_scale(batchsize, seq_length);

    tensor = handle->global_malloc_manage_float.get_new_head_point(
                                         batchsize *
                                         seq_length *
                                         handle->hidden_size);
    pooled_output = handle->global_malloc_manage_float.get_new_head_point(
                                         batchsize * 
                                         handle->hidden_size);

    HostApplyEmbeddings<T>(handle, tensor, words, token_types, attention_mask);
    
    for(int i = 0; i < 1; i++){ //handle->num_hidden_layers
        BERT_Layer<T>(handle, tensor, i, attention_mask);
        
    }

    BERT_Pooler<T>(handle, tensor, pooled_output);
    //dump_tensor<T>(std::string("pooler_output"), tensor, batchsize, handle->hidden_size);
}

Retval BERT_Inference (global_manager * handle,
                    int* words, 
                    int* token_types, 
                    int batchsize, 
                    int seq_length, 
                    int* attention_mask) {
    Retval ret;

    handle->set_scale(batchsize, seq_length);
    handle->reset();

    //std::cout<<"INFO: BERT_Inference for Batchsize="
    //      <<batchsize<<" and Seq_Length="<<seq_length<<std::endl;
   
    ret.tensor = handle->global_malloc_manage_float.get_new_head_point(
                                         batchsize *
                                         seq_length *
                                         handle->hidden_size);
    ret.pooled_output = handle->global_malloc_manage_float.get_new_head_point(
                                         batchsize * 
                                         handle->hidden_size);

    HostApplyEmbeddings<float>(handle, ret.tensor, words, token_types, attention_mask);
    //dump_tensor<float>(std::string("embedding_out"), ret.tensor, batchsize, seq_length, handle->hidden_size);

    for(int i = 0; i < handle->num_hidden_layers; i++){ //handle->num_hidden_layers
        BERT_Layer<float>(handle, ret.tensor, i, attention_mask);
        //dump_tensor<float>(std::string("layer_output_"+std::to_string(i)), ret.tensor, batchsize, seq_length, handle->hidden_size);
    }

    BERT_Pooler<float>(handle, ret.tensor, ret.pooled_output);
    //dump_tensor<float>(std::string("pooler_output"), ret.pooled_output, batchsize, handle->hidden_size);

    return ret;
}

template<typename T>
T* classify_inference(global_manager * handle, 
            T* pooled_output, 
            size_t num_classes) {
    size_t batchsize = handle->batchsize;
    size_t hidden_size = handle->hidden_size;
    T* output;
    output = handle->global_malloc_manage_float.get_new_head_point(batchsize*num_classes);

    std::vector<std::string> keys = {"classifier_kernel"};
    T* kernel = look_up_tts(handle->tts, keys)->gpu_mem;
    keys = {"classifier_bias"};
    T* bias = look_up_tts(handle->tts, keys)->gpu_mem;
    Linear<T>(handle, 
              output, 
              pooled_output, 
              kernel, 
              bias, 
              batchsize, 
              hidden_size, 
              num_classes, 
              false);
      
    HostApplySoftmax(handle, output, batchsize, num_classes);

    return output;
}

template
float* classify_inference<float>(global_manager * handle, 
                                float* pooled_output, 
                                size_t num_classes);
