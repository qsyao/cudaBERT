#ifndef BERT_CUDA_ELEMENTWISE
#define BERT_CUDA_ELEMENTWISE

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include "math.h"

#include "../utils/common.h"

template <typename T> 
__global__ void MemoryCpyLinear(T* out, T* in, int max, int warpsize) ;

template <typename T> 
__device__ void BertTranspose (T* out, 
                              T *in, 
                              size_t batchsize, 
                              size_t seq_length, 
                              size_t num_heads, 
                              long total_length,
                              bool muti_head);

template <typename T>
__global__ void FusionTranspose (T* out,
                                T* in, 
                                int num, 
                                size_t batchsize, 
                                size_t seq_length,
                                size_t num_heads,
                                long total_length,
                                long total, 
                                bool muti_head) ;

template <typename T>
__global__ void BertDiv (T* tensor, float number, int max_num) ;

template <typename T>
__global__ void BertAdd (T* tensor, T* add, int max_num);

template <typename T>
__global__ void Attention_Mask_Add_Merge_div_only_Add (
                                T* tensor, 
                                int* mask, 
                                float number, 
                                int max_num, 
                                int batchsize, 
                                int seq_length) ;

template <typename T>
__global__ void Attention_Mask_Add_Merge_div_only_div (
                                  T* tensor, 
                                  int* mask, 
                                  float number, 
                                  int max_num, 
                                  int batchsize, 
                                  int seq_length);

template <typename T>
__global__ void BertGelu (T* tensor,  int max_num) ;

template <typename T>
__global__ void BertTanh (T* tensor,  int max_num) ;

#endif
