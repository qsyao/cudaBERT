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

        Query_Key* qk = new Query_Key(handle);
        query_key.push_back(qk);

        Prob_Value* pv = new Prob_Value(handle);
        prob_value.push_back(pv);

        op_BatchedLinear* batchlinear = new op_BatchedLinear(
                                        num_layer + "attention_self_query_kernel",
                                        num_layer + "attention_self_query_bias",
                                        num_layer + "attention_self_key_kernel",
                                        num_layer + "attention_self_key_bias",
                                        num_layer + "attention_self_value_kernel",
                                        num_layer + "attention_self_value_bias",
                                        handle);
        batched_linear.push_back(batchlinear);

        op_Gelu* op_gelu = new op_Gelu(handle);
        gelu.push_back(op_gelu);
    }

    pooler_linear = new op_Linear(  "pooler_dense_kernel",
                                    "pooler_dense_bias",
                                    handle);
    
    classify_linear = new op_Linear("classifier_kernel",
                                    "classifier_bias",
                                    handle);

    loss = new op_CrossEntropyLoss(handle);

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

    int num_words = batchsize * seq_length;
    int *word_gpu, *token_type_gpu, *positions_gpu, *mask_gpu;

    int positions[num_words];
    for( int i = 0; i < num_words; i++){
        positions[i] = i % seq_length;
    }
    
    int* host_input_package;
    checkCudaErrors(cudaMallocHost((void **)&host_input_package, 4*num_words*sizeof(int)));
    memcpy(host_input_package, words, num_words*sizeof(int));
    memcpy(host_input_package + num_words, token_type, num_words*sizeof(int));
    memcpy(host_input_package + 2*num_words, positions, num_words*sizeof(int));

    word_gpu = handle->global_malloc_manage_int.get_new_head_point(num_words);
    token_type_gpu = handle->global_malloc_manage_int.get_new_head_point(num_words);
    positions_gpu = handle->global_malloc_manage_int.get_new_head_point(num_words);
    if(attention_mask != nullptr){
        mask_gpu = handle->global_malloc_manage_int.get_new_head_point(num_words);
        memcpy(host_input_package + 3*num_words, attention_mask, num_words*sizeof(int));
        checkCudaErrors(cudaMemcpyAsync(word_gpu, 
                                        host_input_package, 
                                        4*num_words*sizeof(int), 
                                        cudaMemcpyHostToDevice));
        attention_mask = mask_gpu;
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(word_gpu, 
                                        host_input_package, 
                                        3*num_words*sizeof(int), 
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

        float *batched_gemm_out;
        batched_linear[i]->forward(batched_gemm_out,
                                   tensor_layer,
                                   batchsize * seq_length,
                                   hidden_size,
                                   hidden_size);

        float *query, *key, *val;
        query = batched_gemm_out;
        key = query + total_length;
        val = key + total_length;

        float *query_key_gemm;
        query_key[i]->forward(
                            query,
                            key,
                            1.0 / 8.0,
                            query_key_gemm);
        
        softmax[i]->forward(handle,
                            query_key_gemm,
                            batchsize * num_attention_heads * seq_length,
                            seq_length,
                            attention_mask);
        
        float* attention;
        prob_value[i]->forward(
                             query_key_gemm,
                             val,
                             attention);
        
        attention_linear[i]->forward(temp,
                                    attention,
                                    num_words,
                                    hidden_size,
                                    hidden_size);
        attention = temp;
        
        attention_layernorm[i]->forward(tensor_layer,
                                        tensor_layer,
                                        num_words,
                                        hidden_size,
                                        attention);
    
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

        cudaEventRecord(handle->layer_compute_done, handle->cal_stream);
        cudaEventSynchronize(handle->layer_compute_done);
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

float *bert::classify_inference(int *classes, float *pooler_out, size_t num_classes) {
    float *loss_out;
    float *classify_out;
    classify_linear->forward(classify_out,
                             pooler_out,
                             handle->batchsize,
                             handle->hidden_size,
                             num_classes);
//    debug_tensor_gpu<float>(std::string("classify_out"), classify_out, 2, 2, handle->batchsize);

    int *calsses_gpu;
    calsses_gpu = handle->global_malloc_manage_int.get_new_head_point(handle->hidden_size);
    checkCudaErrors(cudaMemcpyAsync(calsses_gpu, classes, handle->hidden_size * sizeof(int), cudaMemcpyHostToDevice));

    loss->forward(loss_out, classify_out, calsses_gpu, handle->batchsize, num_classes);
//    debug_tensor_gpu<float>(std::string("CrossEntropyLoss_output"), loss_out, handle->batchsize + 1,
//                            handle->batchsize + 1);

    return loss_out;
}

void bert::classify_inference_backward(int *classes, size_t num_classes) {
    int *calsses_gpu;
    calsses_gpu = handle->global_malloc_manage_int.get_new_head_point(handle->hidden_size);
    checkCudaErrors(cudaMemcpyAsync(calsses_gpu, classes, handle->hidden_size * sizeof(int), cudaMemcpyHostToDevice));

    float *dout_gpu;
    dout_gpu = handle->global_malloc_manage_float.get_new_head_point(1);
    float *dout = (float *) malloc(sizeof(float));
    dout[0] = 1.0;
    checkCudaErrors(cudaMemcpyAsync(dout_gpu, dout, sizeof(float), cudaMemcpyHostToDevice));

    loss->backward(dout_gpu, handle->batchsize, num_classes, calsses_gpu);
//    debug_tensor_gpu<float>(std::string("Grid CrossEntropyLoss_output"), loss->grad_input, n2, n2, n1);

    // debug_tensor_gpu<float>(std::string("classify_linear->stored_input"), classify_linear->stored_input, 768, 768, 2);
    classify_linear->backward(loss->grad_input, handle->batchsize,
                              handle->hidden_size,
                              num_classes);
//    debug_tensor_gpu<float>(std::string("grad_input"), classify_linear->grad_input, k, k, n);
//    debug_tensor_gpu<float>(std::string("grad_kernel"), grad_kernel, m, m, k);
//    debug_tensor_gpu<float>(std::string("grad_bias"), grad_bias, m, m);

    return;
}