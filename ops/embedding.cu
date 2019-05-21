#include "embedding.cuh"

template<typename T>
__global__
void DeviceApplyEmbeddings(const int *words,
                           int *token_type,
                           int *positions,
                           int total_length,
                           T *output,
                           T *embedding_words,
                           T *embedding_token_types,
                           T *embedding_positions) {
    int warp_size = blockDim.x;
    for (int index = blockIdx.x; index < total_length; index += gridDim.x) {
        int curnIdx = index * warp_size + threadIdx.x;
        output[curnIdx] = embedding_words[warp_size * words[index] + threadIdx.x] +
                          embedding_token_types[warp_size * token_type[index] + threadIdx.x] +
                          embedding_positions[warp_size * positions[index] + threadIdx.x];
    }
    __syncthreads();
}

template<typename T>
void Embedding::forward(T *&output,
                        int *words,
                        int *token_type,
                        int *position) {
    word_input = words;
    token_type_input = token_type;
    position_input = position;
    // MemcpyHostToDevice for inputs
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;

    int total_words = batchsize * seq_length;

    output = handle->global_malloc_manage_float.get_new_head_point(
            batchsize *
            seq_length *
            hidden_size);

    dim3 threads(hidden_size, 1, 1);
    dim3 blocks(min(65536, total_words), 1, 1);
    DeviceApplyEmbeddings << < blocks, threads, 0, handle->cal_stream >> > (
            words,
                    token_type,
                    position,
                    total_words,
                    output,
                    word_embedding,
                    token_type_embedding,
                    position_embedding);
    //debug_tensor_gpu<float>(std::string("after embedding add : "), output, 10, handle->hidden_size, 5);

    layernorm->forward(output,
                       output,
                       total_words,
                       hidden_size);

    //debug_tensor_gpu<float>(std::string("look_up_embedding on GPU"), output, 10, 768, 11*batchsize);
}

template
void Embedding::forward<float>(
        float *&output,
        int *words,
        int *token_type,
        int *position);

template<typename T>
__global__
void DeviceApplyEmbeddingsGrad(const int *words,
                               const int *token_type,
                               const int *positions,
                               int total_length,
                               T *dout,
                               T *grad_embedding_words,
                               T *grad_embedding_token_types,
                               T *grad_embedding_positions) {
    int warp_size = blockDim.x;
    for (int index = blockIdx.x; index < total_length; index += gridDim.x) {
        int curnIdx = index * warp_size + threadIdx.x;
        grad_embedding_words[warp_size * words[index] + threadIdx.x] += dout[curnIdx];
        grad_embedding_token_types[warp_size * token_type[index] + threadIdx.x] += dout[curnIdx];
        grad_embedding_positions[warp_size * positions[index] + threadIdx.x] += dout[curnIdx];
    }
    __syncthreads();
}

void Embedding::update_weights() {
    if (handle->update_learning_rate) {
        learning_rate = handle->learning_rate;
    }
    if (handle->optim_method == "sgd") {
        apply_sgd_running_time(word_embedding, grad_word_embedding, len_word_embedding, learning_rate, handle);
        apply_sgd_running_time(position_embedding, grad_position_embedding, len_position_embedding, learning_rate,
                               handle);
        apply_sgd_running_time(token_type_embedding, grad_token_type_embedding, len_token_type_embedding, learning_rate,
                               handle);
    } else if (handle->optim_method == "adam") {
        apply_adam_running_time(word_embedding, grad_word_embedding, len_word_embedding, word_m_t, word_v_t,
                                beta_1_t,
                                beta_2_t, handle, learning_rate, weight_decay_rate, beta_1,
                                beta_2, adam_epsilon, step);

        apply_adam_running_time(position_embedding, grad_position_embedding, len_position_embedding, position_m_t,
                                position_v_t, beta_1_t,
                                beta_2_t, handle, learning_rate, weight_decay_rate, beta_1,
                                beta_2, adam_epsilon, step);

        apply_adam_running_time(token_type_embedding, grad_token_type_embedding, len_token_type_embedding,
                                token_type_m_t, token_type_v_t, beta_1_t,
                                beta_2_t, handle, learning_rate, weight_decay_rate, beta_1,
                                beta_2, adam_epsilon, step);

        beta_1_t = beta_1_t * beta_1;
        beta_2_t = beta_2_t * beta_2;
        step += 1;
    } else if (handle->optim_method == "momentum") {
        apply_momentum_running_time(word_embedding, grad_word_embedding, len_word_embedding, momentum_word_v,
                                    learning_rate, momentum_beta, handle, step);
        apply_momentum_running_time(position_embedding, grad_position_embedding, len_position_embedding,
                                    momentum_position_v, learning_rate, momentum_beta, handle, step);
        apply_momentum_running_time(token_type_embedding, grad_token_type_embedding, len_token_type_embedding,
                                    momentum_token_type_v, learning_rate, momentum_beta, handle, step);

        step += 1;
    }

}

template<typename T>
void Embedding::backward(T *dout) {
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;

    debug_tensor_gpu<float>(std::string("layernorm->gamma: "), layernorm->gamma, 3, handle->hidden_size, 1);
    debug_tensor_gpu<float>(std::string("dout"), dout, 3, hidden_size, batchsize * seq_length);
    layernorm->backward(dout, batchsize * seq_length, hidden_size);

    debug_tensor_gpu<float>(std::string("layernorm->grad_gamma"), layernorm->grad_gamma, hidden_size, hidden_size);
    debug_tensor_gpu<float>(std::string("layernorm->grad_input"), layernorm->grad_input, 3, hidden_size, batchsize * seq_length);

//    int total_words = batchsize * seq_length;
//
//    grad_word_embedding = handle->global_malloc_manage_float.get_new_head_point(len_word_embedding);
//    cudaMemset(grad_word_embedding, 0, len_word_embedding * sizeof(float));
//
//    grad_position_embedding = handle->global_malloc_manage_float.get_new_head_point(len_position_embedding);
//    cudaMemset(grad_position_embedding, 0, len_position_embedding * sizeof(float));
//
//    grad_token_type_embedding = handle->global_malloc_manage_float.get_new_head_point(len_token_type_embedding);
//    cudaMemset(grad_token_type_embedding, 0, len_token_type_embedding * sizeof(float));
//
//    dim3 threads(hidden_size, 1, 1);
//    dim3 blocks(min(65536, total_words), 1, 1);
//
////    debug_tensor_gpu<float>(std::string("grad_token_type_embedding"), grad_token_type_embedding, 3, hidden_size, 2);
//
//    DeviceApplyEmbeddingsGrad << < blocks, threads, 0, handle->cal_stream >> > (
//            word_input,
//                    token_type_input,
//                    position_input,
//                    total_words,
//                    layernorm->grad_input,
//                    grad_word_embedding,
//                    grad_token_type_embedding,
//                    grad_position_embedding);
//
//
////    debug_tensor_gpu<float>(std::string("grad_token_type_embedding"), grad_token_type_embedding, 3, hidden_size, 2);
//
//    if (handle->optim_running_time)
//        update_weights();
}

template
void Embedding::backward(float *dout);