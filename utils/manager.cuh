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
        global_handle (bool BERT_Large=false, std::string dir = "", bool optimRunningTime = true, bool isTrain = false, int num_classes = 2);

        ~global_handle();

        void set_optim_sgd(float lr = 0.001);

        void set_optim_adam(float learning_rate = 0.001, float weight_decay_rate = 0.0, float beta_1 = 0.9,
                            float beta_2 = 0.999, float epsilon = 1e-8);

        void set_optim_momentum(float lr = 0.001, float beat = 0.9);

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

        std::string dir_npy = "model_npy/base_uncased";
        size_t hidden_size = 768;
        size_t num_hidden_layers = 12;
        size_t num_attention_heads = 12;
        size_t intermediate_size = 3072;

        size_t batchsize;
        size_t seq_length;
        size_t max_seq_length = 512;
        size_t max_mem_size = 200 * 512;
        float learning_rate = 0.001;
        float hidden_dropout_prob = 0.1;
        float attention_probs_dropout_prob = 0.1;
        bool update_learning_rate = false;
        std::string optim_method = "sgd";
        bool optim_running_time = true;
        bool is_train = false;
        float momentum_beta;
        float beta_1;
        float beta_2;
        float weight_decay_rate;
        float epsilon;
        int num_classes;

        malloc_manage<float> global_malloc_manage_float;
        malloc_manage<int> global_malloc_manage_int;

        std::vector<tagged_tensor *> tts;

        cudaStream_t cal_stream;
        cudaStream_t copy_stream;
        
        cudaEvent_t layer_compute_done;
};

#endif
