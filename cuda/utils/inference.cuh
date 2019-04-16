#ifndef CUDA_BERT_INFERENCE
#define CUDA_BERT_INFERENCE

#include "../ops/layernorm.cuh"
#include "../ops/linear.cuh"
#include "../ops/embedding.cuh"
#include "../ops/matmul.cuh"
#include "../ops/softmax.cuh"
#include "../utils/common.h"
#include "../ops/elementwise.cuh"
#include "../utils/manager.cuh"

template <typename T>
void BERT_Inference (global_manager *handle, 
                    T* &tensor,  
                    T* &pooled_output,
                    int* words, 
                    int* token_types, 
                    size_t batchsize, 
                    size_t seq_length, 
                    int* attention_mask=nullptr) ;

extern "C"
Retval BERT_Inference (global_manager * handle,
                    int* words, 
                    int* token_types, 
                    int batchsize, 
                    int seq_length, 
                    int* attention_mask=nullptr);

template<typename T>
T* classify_inference(global_manager * handle, 
            T* pooled_output, 
            size_t num_classes);

#endif
