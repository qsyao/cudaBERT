#ifndef CUDA_BERT_EMBEDDING
#define CUDA_BERT_EMBEDDING

#include "op_kernel.cuh"
#include "layernorm.cuh"

class Embedding : public op_kernel{
  public:

    Embedding(global_handle* handle)
                   : op_kernel(handle){
        layernorm = new op_LayerNorm("embeddings_LayerNorm_gamma",
                                     "embeddings_LayerNorm_beta",
                                      handle);
        std::vector<std::string> keys = {"embeddings", "word"};
        word_embedding = look_up_tts(handle->tts, keys)->gpu_mem;
        keys = {"embeddings", "position"};
        position_embedding = look_up_tts(handle->tts, keys)->gpu_mem;
        keys = {"embeddings", "token_type"};
        token_type_embedding = look_up_tts(handle->tts, keys)->gpu_mem; 
}

    ~Embedding();

    template <typename T>
    void forward (
                T* &output,
                int *words,
                int *token_type, 
                int *position);
    
    void backward();

    void update_weights();

  private:
    op_LayerNorm* layernorm;
    float* word_embedding;
    float* position_embedding;
    float* token_type_embedding;
};

#endif
