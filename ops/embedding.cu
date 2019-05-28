#include "embedding.cuh"
#include "shfl.cuh"

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

    float *tmp_output = handle->global_malloc_manage_float.get_new_head_point(
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
                    tmp_output,
                    word_embedding,
                    token_type_embedding,
                    position_embedding);
    //debug_tensor_gpu<float>(std::string("after embedding add : "), output, 10, handle->hidden_size, 5);

    layernorm->forward(output,
                       tmp_output,
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
void DeviceApplyEmbeddingsGrad(T *grad_input,
                                  T *dout,
                                  int n2,
                                  int word_index,
                                  int hidden_size,
                                  int WARP_SIZE,
                                  const int *word_pos) {
    T temp = 0;
    int i1 = blockIdx.x;
    for (int i2 = threadIdx.x; i2 < n2; i2 += blockDim.x) {
        temp += dout[i1 + word_pos[i2] * hidden_size];
    }
    __syncthreads();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        temp += WARP_SHFL_DOWN(temp, offset);

    if (threadIdx.x == 0)
        grad_input[i1 + word_index * hidden_size] = temp;
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
void Embedding::backward(T *dout) {        debug_tensor_gpu<int>(std::string("nmb"), word_input, 128, 128, 8);
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t hidden_size = handle->hidden_size;

//    debug_tensor_gpu<float>(std::string("layernorm->gamma: "), layernorm->gamma, 3, hidden_size, 1);
//    debug_tensor_gpu<float>(std::string("dout"), dout, 3, hidden_size, batchsize * seq_length);
    layernorm->backward(dout, batchsize * seq_length, hidden_size);

//    debug_tensor_gpu<float>(std::string("layernorm->grad_gamma"), layernorm->grad_gamma, 3, hidden_size);
//    debug_tensor_gpu<float>(std::string("layernorm->grad_input"), layernorm->grad_input, 3, hidden_size, batchsize * seq_length);

    int total_words = batchsize * seq_length;

    grad_word_embedding = handle->global_malloc_manage_float.get_new_head_point(len_word_embedding);

    grad_position_embedding = handle->global_malloc_manage_float.get_new_head_point(len_position_embedding);

    grad_token_type_embedding = handle->global_malloc_manage_float.get_new_head_point(len_token_type_embedding);

    {
        int *word_id_use = (int *) malloc(sizeof(int) * len_word_embedding / hidden_size);
        memset(word_id_use, 0, sizeof(int) * len_word_embedding / hidden_size);
        int *word_input_cpu = (int *) malloc(sizeof(int) * total_words);
        checkCudaErrors(cudaMemcpy(word_input_cpu, word_input, sizeof(int) * total_words, cudaMemcpyDeviceToHost));

        int *word_pos = handle->global_malloc_manage_int.get_new_head_point(total_words);
        int *word_pos_cpu = (int *) malloc(sizeof(int) * total_words);
        for (int i = 0; i < total_words; i++) {
            if (word_id_use[word_input_cpu[i]] == 1)
                continue;
            int count = 0, j = i;
            word_id_use[word_input_cpu[i]] = 1;
            while (j < total_words) {
                if (word_input_cpu[i] == word_input_cpu[j])
                    word_pos_cpu[count++] = j;
                j++;
            }
            checkCudaErrors(cudaMemcpy(word_pos, word_pos_cpu, sizeof(int) * count, cudaMemcpyHostToDevice));
            dim3 threads(min(32, count), 1, 1);
            dim3 blocks(hidden_size, 1, 1);
            DeviceApplyEmbeddingsGrad << < blocks, threads, 0, handle->cal_stream >> > (
                    grad_word_embedding,
                            layernorm->grad_input,
                            count,
                            word_input_cpu[i],
                            hidden_size,
                            32,
                            word_pos
            );
        }
    }
    {
        int *token_type_id_use = (int *) malloc(sizeof(int) * len_token_type_embedding / hidden_size);
        memset(token_type_id_use, 0, sizeof(int) * len_token_type_embedding / hidden_size);
        int *token_type_input_cpu = (int *) malloc(sizeof(int) * total_words);
        checkCudaErrors(cudaMemcpy(token_type_input_cpu, token_type_input, sizeof(int) * total_words, cudaMemcpyDeviceToHost));

        int *token_type_pos = handle->global_malloc_manage_int.get_new_head_point(total_words);
        int *token_type_pos_cpu = (int *) malloc(sizeof(int) * total_words);
        for (int i = 0; i < total_words; i++) {
            if (token_type_id_use[token_type_input_cpu[i]] == 1)
                continue;
            int count = 0, j = i;
            token_type_id_use[token_type_input_cpu[i]] = 1;
            while (j < total_words) {
                if (token_type_input_cpu[i] == token_type_input_cpu[j])
                    token_type_pos_cpu[count++] = j;
                j++;
            }
            checkCudaErrors(cudaMemcpy(token_type_pos, token_type_pos_cpu, sizeof(int) * count, cudaMemcpyHostToDevice));
            dim3 threads(min(32, count), 1, 1);
            dim3 blocks(hidden_size, 1, 1);
            DeviceApplyEmbeddingsGrad << < blocks, threads, 0, handle->cal_stream >> > (
                    grad_token_type_embedding,
                            layernorm->grad_input,
                            count,
                            token_type_input_cpu[i],
                            hidden_size,
                            32,
                            token_type_pos
            );
        }
//        debug_tensor_gpu<float>(std::string("grad_token_type_embedding"), grad_token_type_embedding, 3, hidden_size, 2);
    }

    {
        int *position_id_use = (int *) malloc(sizeof(int) * len_position_embedding / hidden_size);
        memset(position_id_use, 0, sizeof(int) * len_position_embedding / hidden_size);
        int *position_input_cpu = (int *) malloc(sizeof(int) * total_words);
        checkCudaErrors(cudaMemcpy(position_input_cpu, position_input, sizeof(int) * total_words, cudaMemcpyDeviceToHost));

        int *position_pos = handle->global_malloc_manage_int.get_new_head_point(total_words);
        int *position_pos_cpu = (int *) malloc(sizeof(int) * total_words);
        for (int i = 0; i < total_words; i++) {
            if (position_id_use[position_input_cpu[i]] == 1)
                continue;
            int count = 0, j = i;
            position_id_use[position_input_cpu[i]] = 1;
            while (j < total_words) {
                if (position_input_cpu[i] == position_input_cpu[j])
                    position_pos_cpu[count++] = j;
                j++;
            }
            checkCudaErrors(cudaMemcpy(position_pos, position_pos_cpu, sizeof(int) * count, cudaMemcpyHostToDevice));
            dim3 threads(min(32, count), 1, 1);
            dim3 blocks(hidden_size, 1, 1);
            DeviceApplyEmbeddingsGrad << < blocks, threads, 0, handle->cal_stream >> > (
                    grad_position_embedding,
                            layernorm->grad_input,
                            count,
                            position_input_cpu[i],
                            hidden_size,
                            32,
                            position_pos
            );
        }
//        debug_tensor_gpu<float>(std::string("grad_position_embedding"), grad_position_embedding, 3, hidden_size, 3);
    }

    if (handle->optim_running_time)
        update_weights();
}

template
void Embedding::backward(float *dout);
