#ifndef CUDA_BERT_MANAGER
#define CUDA_BERT_MANAGER

#include<string>
#include<vector>
#include "cuda_runtime.h"
#include <cublas_v2.h>

#include "common.h"
#include "load_model.h"
#include "../ops/elementwise.cuh"

extern "C"
class global_manager {
    public:
        global_manager (bool BERT_Large=false, std::string dir = "");

        ~global_manager();

        void init_cudamemory(int batchsize, int seq_length);

        void prepare_linear(global_manager *handle,
                    std::vector<tagged_tensor *>& tts, 
                    dict_weights &weights);

        void set_scale(size_t input_batchsize, size_t input_seq_length);

        void reset(){
            global_malloc_manage_float.set_head_zero();
            global_malloc_manage_int.set_head_zero();
        }

        cudaStream_t get_cal_stream(){
            return cal_stream;
        }

        cudaStream_t get_copy_stream(){
            return copy_stream;
        }

        std::vector<tagged_tensor *> tts;
        dict_weights weights;
        cublasHandle_t handle;

        cudaEvent_t copy_event;

        std::string dir_npy = "model_npy/base_uncased";
        size_t hidden_size = 768;
        size_t num_hidden_layers = 12;
        size_t num_attention_heads = 12;
        size_t intermediate_size = 3072;

        size_t batchsize;
        size_t seq_length;
        size_t max_batchsize = 128;
        size_t max_mem_size = 80 * 1000;

        malloc_manage<float> global_malloc_manage_float;
        malloc_manage<int> global_malloc_manage_int;

    private:

        cudaStream_t cal_stream;
        cudaStream_t copy_stream;
};

#endif
