#ifndef CUDA_BERT_EMBEDDING
#define CUDA_BERT_EMBEDDING

#include "cuda_runtime.h"

#include "../utils/common.h"
#include "layernorm.cu"
#include "../utils/load_model.h"

template <typename T>
void HostApplyEmbeddings (global_manager *handle, 
                         T* &output,
                         int *words, 
                         int *token_type, 
                         int* &attention_mask = nullptr);

#endif
