#include "bert.cuh"
#include "load_model.h"

bert::bert (bool BERT_Large, int num_gpu, std::string dir) {
    checkCudaErrors(cudaSetDevice(num_gpu));
    handle = new global_handle(BERT_Large, dir);
    init_ops();
}

void bert::init_ops(){

    for(int i = 0; i < handle->num_hidden_layers; i++){
        std::string num_layer = "_" + std::to_string(i) + "_";

        op_LayerNorm* layernorm = new op_LayerNorm(num_layer + "attention_output_LayerNorm_gamma",
                                                num_layer + "attention_output_LayerNorm_beta",
                                                handle);
        attention_layernorm.push_back(layernorm);

        layernorm = new op_LayerNorm(num_layer + "output_LayerNorm_gamma",
                                    num_layer + "output_LayerNorm_beta",
                                    handle);
        output_layernorm.push_back(layernorm);

        op_SoftMax* Softmax = new op_SoftMax(handle);
        softmax.push_back(Softmax);

        op_Linear* linear = new op_Linear(num_layer + "attention_output_dense_kernel",
                                          num_layer + "attention_output_dense_bias",
                                           handle);
        attention_linear.push_back(linear);
        
        linear = new op_Linear(num_layer + "intermediate_dense_kernel",
                                num_layer + "intermediate_dense_bias",
                                handle);
        intermediate_linear.push_back(linear);

        linear = new op_Linear(num_layer + "output_dense_kernel",
                                num_layer + "output_dense_bias",
                                handle);
        output_linear.push_back(linear);

        op_Batch_Matmul* batchgemm = new op_Batch_Matmul(handle);
        query_key.push_back(batchgemm);

        batchgemm = new op_Batch_Matmul(handle);
        head_value.push_back(batchgemm);

        op_BatchedLinear* batchlinear = new op_BatchedLinear(
                                        num_layer + "attention_self_query_kernel",
                                        num_layer + "attention_self_query_bias",
                                        num_layer + "attention_self_key_kernel",
                                        num_layer + "attention_self_key_bias",
                                        num_layer + "attention_self_value_kernel",
                                        num_layer + "attention_self_value_bias",
                                        handle);
        batched_linear.push_back(batchlinear);

        op_FusionTranspose* trans = new op_FusionTranspose(handle);
        split_heads.push_back(trans);

        trans = new op_FusionTranspose(handle);
        merge_heads.push_back(trans);

        op_Mask_Add* op_mask = new op_Mask_Add(handle);
        mask.push_back(op_mask);

        op_Gelu* op_gelu = new op_Gelu(handle);
        gelu.push_back(op_gelu);
    }

    pooler_linear = new op_Linear(  "pooler_dense_kernel",
                                    "pooler_dense_bias",
                                    handle);
    
    classify_linear = new op_Linear("classifier_kernel",
                                    "classifier_bias",
                                    handle);
    
    classify_softmax = new op_SoftMax(handle);

    embedding = new Embedding(handle);

    op_tanh = new op_Tanh(handle);
}

void bert::copy_inputs( int* &words, 
                        int* &token_type,
                        int* &position,
                        int* &attention_mask){
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;

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
        checkCudaErrors(cudaMemcpyAsync(word_gpu, 
                                        host_input_package, 
                                        4*total_length*sizeof(int), 
                                        cudaMemcpyHostToDevice));
        attention_mask = mask_gpu;
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(word_gpu, 
                                        host_input_package, 
                                        3*total_length*sizeof(int), 
                                        cudaMemcpyHostToDevice));
    }
    cudaFreeHost(host_input_package);
    words = word_gpu;
    token_type = token_type_gpu;
    position = positions_gpu;
}

void bert::BERT_Inference (
                    int* words, 
                    int* token_types, 
                    size_t batchsize, 
                    size_t seq_length, 
                    int* attention_mask){

    size_t hidden_size = handle->hidden_size;
    size_t total_length = batchsize * seq_length * hidden_size;
    size_t num_words = batchsize * seq_length;
    size_t num_attention_heads= handle->num_attention_heads;
    size_t intermediate_size = handle->intermediate_size;

    handle->set_scale(batchsize, seq_length);
    handle->reset();

    int* positions;
    copy_inputs(words, token_types, positions, attention_mask);

    float *embedding_out;

    embedding->forward(embedding_out, words, token_types, positions);

    float *tensor_layer = embedding_out, *temp;

    for(int i = 0; i < handle->num_hidden_layers; i++){

        handle->global_malloc_manage_float.record_layer_start();
        
        // start of Attention

        float *batched_gemm_out, *split_heads_out;
        batched_linear[i]->forward(batched_gemm_out,
                                   tensor_layer,
                                   batchsize * seq_length,
                                   hidden_size,
                                   hidden_size);
        
        split_heads[i]->forward(split_heads_out, batched_gemm_out, 3, true);

        float *head_query, *head_key, *head_val;
        head_query = split_heads_out;
        head_key = head_query + total_length;
        head_val = head_key + total_length;


        float *query_key_gemm;
        query_key[i]->forward(batchsize * num_attention_heads,
                            seq_length,
                            hidden_size / num_attention_heads,
                            seq_length,
                            head_query,
                            head_key,
                            query_key_gemm,
                            false,
                            true);
        
        mask[i]->forward(query_key_gemm, attention_mask, 8.0);
        
        softmax[i]->forward(query_key_gemm,
                            batchsize * num_attention_heads * seq_length,
                            seq_length);
        
        float* attention;
        head_value[i]->forward(batchsize * num_attention_heads,
                             seq_length,
                             seq_length,
                             hidden_size / num_attention_heads,
                             query_key_gemm,
                             head_val,
                             attention,
                             false,
                             false);
        
        float *merge_heads_out;
        merge_heads[i]->forward(merge_heads_out, attention, 1, false);
        
        attention_linear[i]->forward(temp,
                                     merge_heads_out,
                                     num_words,
                                     hidden_size,
                                     hidden_size);
        merge_heads_out = temp;
        
        attention_layernorm[i]->forward(tensor_layer,
                                        tensor_layer,
                                        num_words,
                                        hidden_size,
                                        merge_heads_out);
                
        // End of Attention
        // Start of Intermediate

        float* intermediate_out;
        intermediate_linear[i]->forward(intermediate_out,
                                        tensor_layer,
                                        num_words,
                                        hidden_size,
                                        intermediate_size);

        gelu[i]->forward(intermediate_out, num_words * intermediate_size);

        // End of Intermedaite
        // Start of Output

        float* output_out;
        output_linear[i]->forward(output_out,
                                  intermediate_out,
                                  num_words,
                                  intermediate_size,
                                  hidden_size);

        output_layernorm[i]->forward(tensor_layer,
                                     tensor_layer,
                                     num_words,
                                     hidden_size,
                                     output_out);

        cudaStreamSynchronize(handle->cal_stream);
        handle->global_malloc_manage_float.reuse_layer_mem();
        //  Layer End
    }
    // Pooler Start
    float* first_token, *pooler_out;
    copy_pooler(first_token, tensor_layer, handle);

    pooler_linear->forward(pooler_out,
                           first_token,
                           batchsize,
                           hidden_size,
                           hidden_size);
    
    op_tanh->forward(pooler_out, batchsize * hidden_size);
    // Pooler End
    
    ret.tensor = tensor_layer;
    ret.pooled_output = pooler_out;
}

float* bert::classify_inference(float* pooler_out, size_t num_classes){
    float* classify_out;
    classify_linear->forward(classify_out,
                             pooler_out,
                             handle->batchsize,
                             handle->hidden_size,
                             num_classes);
    
    classify_softmax->forward(classify_out, handle->batchsize, num_classes);

    return classify_out;
}
