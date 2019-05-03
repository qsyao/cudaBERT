#ifndef CUDA_BERT_MANAGER
#define CUDA_BERT_MANAGER

#include<string>
#include<vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "load_model.h"
#include "common.h"

extern "C"
class global_handle {
    public:
        global_handle (bool BERT_Large=false, std::string dir = "", float lr = 0.000001, std::string optim = "sgd", bool optimRunningTime = true);

        ~global_handle();

        void init_cudamemory(int batchsize, int seq_length);

        void prepare_linear(global_handle *handle,
                    std::vector<tagged_tensor *>& tts, 
                    dict_weights &weights);

        void set_scale(size_t input_batchsize, size_t input_seq_length);

        void reset(){
            global_malloc_manage_float.set_head_zero();
            global_malloc_manage_int.set_head_zero();
        }

        cublasHandle_t handle;

        cudaEvent_t copy_event;

        std::string dir_npy = "/home/wenxh/zyc/model_npy/base_uncased";
        size_t hidden_size = 768;
        size_t num_hidden_layers = 12;
        size_t num_attention_heads = 12;
        size_t intermediate_size = 3072;

        size_t batchsize;
        size_t seq_length;
        size_t max_seq_length = 512;
        size_t max_mem_size = 200 * 512;
        float learning_rate = 0.000001;
        std::string optim_method = "sgd";
        bool optim_running_time = true;

        malloc_manage<float> global_malloc_manage_float;
        malloc_manage<int> global_malloc_manage_int;

        std::vector<tagged_tensor *> tts;

        cudaStream_t cal_stream;
        cudaStream_t copy_stream;
        
        cudaEvent_t layer_compute_done;
};

#endif
