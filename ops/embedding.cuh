#ifndef CUDA_BERT_EMBEDDING
#define CUDA_BERT_EMBEDDING

#include "op_kernel.cuh"
#include "layernorm.cuh"

class Embedding : public op_kernel {
public:

    Embedding(global_handle *handle)
            : op_kernel(handle) {
        layernorm = new op_LayerNorm("embeddings_LayerNorm_gamma",
                                     "embeddings_LayerNorm_beta",
                                     handle, handle->hidden_size);
        std::vector <std::string> keys = {"embeddings", "word"};
        word_embedding = look_up_tts(handle->tts, keys)->gpu_mem;
        len_word_embedding = look_up_tts(handle->tts, keys)->num_elements;

        keys = {"embeddings", "position"};
        position_embedding = look_up_tts(handle->tts, keys)->gpu_mem;
        len_position_embedding = look_up_tts(handle->tts, keys)->num_elements;

        keys = {"embeddings", "token_type"};
        token_type_embedding = look_up_tts(handle->tts, keys)->gpu_mem;
        len_token_type_embedding = look_up_tts(handle->tts, keys)->num_elements;

        if (handle->is_train) {
            if (handle->optim_method == "sgd") {
                learning_rate = handle->learning_rate;
            } else if (handle->optim_method == "adam") {
                learning_rate = handle->learning_rate;
                weight_decay_rate = handle->weight_decay_rate;
                beta_1 = handle->beta_1;
                beta_2 = handle->beta_2;
                beta_1_t = 1.0;
                beta_2_t = 1.0;
                word_m_t = handle->global_malloc_manage_float.get_new_head_point(len_word_embedding);
                word_v_t = handle->global_malloc_manage_float.get_new_head_point(len_word_embedding);

                position_m_t = handle->global_malloc_manage_float.get_new_head_point(len_position_embedding);
                position_v_t = handle->global_malloc_manage_float.get_new_head_point(len_position_embedding);

                token_type_m_t = handle->global_malloc_manage_float.get_new_head_point(len_token_type_embedding);
                token_type_v_t = handle->global_malloc_manage_float.get_new_head_point(len_token_type_embedding);

                adam_epsilon = handle->epsilon;
                step = 0;
            } else if (handle->optim_method == "momentum") {
                momentum_word_v = handle->global_malloc_manage_float.get_new_head_point(len_word_embedding);
                momentum_position_v = handle->global_malloc_manage_float.get_new_head_point(len_position_embedding);
                momentum_token_type_v = handle->global_malloc_manage_float.get_new_head_point(len_token_type_embedding);

                learning_rate = handle->learning_rate;
                momentum_beta = handle->momentum_beta;
                step = 0;
            }
        }
    }

    ~Embedding();

    template<typename T>
    void forward(
            T *&output,
            int *words,
            int *token_type,
            int *position);

    template<typename T>
    void backward(T *dout);

    void update_weights();

public:
    op_LayerNorm *layernorm;
    float *word_embedding;
    size_t len_word_embedding;
    float *position_embedding;
    size_t len_position_embedding;
    float *token_type_embedding;
    size_t len_token_type_embedding;
    float *grad_word_embedding;
    float *grad_position_embedding;
    float *grad_token_type_embedding;
    int *word_input, *position_input, *token_type_input;

    float learning_rate;
    float *momentum_word_v;
    float *momentum_position_v;
    float *momentum_token_type_v;
    float *word_m_t, *word_v_t;
    float *position_m_t, *position_v_t;
    float *token_type_m_t, *token_type_v_t;
    float beta_1, beta_2, momentum_beta, weight_decay_rate, adam_epsilon, beta_1_t, beta_2_t;

    int step = 0;
};

#endif
