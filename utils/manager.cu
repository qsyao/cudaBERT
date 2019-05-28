#include "manager.cuh"
#include "load_model.h"

global_handle::global_handle (int max_batchsize,
                              int max_seq_length,
                              bool BERT_Large, 
                              std::string dir, 
                              bool optimRunningTime, 
                              bool isTrain, 
                              int numClasses) :
            max_train_batchsize(max_batchsize),
            max_train_seq_length(max_seq_length) {
    if(BERT_Large){
        dir_npy = "model_npy/large_uncased";
        hidden_size = 1024;
        num_hidden_layers = 24;
        num_attention_heads = 16;
        intermediate_size = 4096;
    }
    if (dir != "") 
        dir_npy = dir;
    optim_running_time = optimRunningTime;
    is_train = isTrain;
    num_classes = numClasses;
    load_from_dir_to_GPU(dir_npy, tts);
    checkError(cublasCreate(&handle), "cublasCreate() error!\n");
    cudaStreamCreate(&cal_stream);
    cudaStreamCreate(&copy_stream);
    cudaEventCreate(&copy_event);
    cudaEventCreate(&layer_compute_done);
    checkError(cublasSetStream(handle, cal_stream), "Set cublas stream Error!\n");
}

void global_handle::set_optim_sgd(float lr) {
    learning_rate = lr;
    optim_method = "sgd";
}

void global_handle::set_optim_adam(float lr, float weightDecayLate, float beta1,
                    float beta2, float eps) {
    learning_rate = lr;
    weight_decay_rate = weightDecayLate;
    beta_1 = beta1;
    beta_2 = beta2;
    epsilon = eps;
    optim_method = "adam";
}

void global_handle::set_optim_momentum(float lr, float beta) {
    learning_rate = lr;
    momentum_beta = beta;
    optim_method = "momentum";
}

global_handle::~global_handle(){
    global_malloc_manage_float.del();
    global_malloc_manage_int.del();
    checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
}

void global_handle::init_cudamemory(int batchsize, int seq_length){
    global_malloc_manage_int.init(batchsize * seq_length * 10 + 2 * batchsize);

    size_t left, total, real_Memcost;
    checkCudaErrors(cudaMemGetInfo(&left, &total));

    left -= 1024 * 1024 * 500;

    std::cout<<"CUDA Memory INFO: Free: "<< left / 1024 / 1024 <<"MB"<<std::endl;
    left = left / sizeof(float);
    
    
    if(!is_train)
        real_Memcost =  batchsize*seq_length*hidden_size + 
                    batchsize*hidden_size*3 +  
                    1 * 
                    (batchsize*seq_length*hidden_size*6 + 
                    batchsize * num_attention_heads * seq_length*seq_length +
                    batchsize*hidden_size*seq_length*3 + 
                    batchsize*seq_length*2 + 
                    batchsize*seq_length * intermediate_size*2 + 
                    batchsize*seq_length*2 +
                    3*hidden_size*hidden_size) + 
                    batchsize*hidden_size*2 +
                    batchsize*seq_length*hidden_size;
    else
        real_Memcost = batchsize * seq_length * hidden_size * 4 +
                       batchsize * seq_length * 2 +
                       num_hidden_layers * (
                           batchsize * seq_length * 4 + 
                           16 * batchsize * seq_length * hidden_size +
                           3 * batchsize * num_attention_heads * seq_length * seq_length +
                           2 * batchsize * seq_length * intermediate_size
                       )  +
                       batchsize * hidden_size * 5 + //forward end
                       batchsize * hidden_size * 5 +
                       hidden_size * 3 + 
                       intermediate_size * hidden_size * 5 +
                       hidden_size * hidden_size * 1 + 
                       batchsize * seq_length * intermediate_size * 2 +
                       batchsize * seq_length * hidden_size * 27 +
                       batchsize * num_attention_heads * seq_length * seq_length * 4 +
                       33000 * hidden_size;
    
    if (real_Memcost > left){
        std::cout<<" Not Enough memory !"<<std::endl;
        assert(false);
    }
    global_malloc_manage_float.init(real_Memcost);
    std::cout<<"Malloc for Batchsize : "<< batchsize
             <<"  Seq_length : " << seq_length 
             <<" Total : "<< real_Memcost * sizeof(float) / 1024 / 1024
             <<"MB" << std::endl;
    
}

void global_handle::set_scale(size_t input_batchsize, size_t input_seq_length){
    batchsize = input_batchsize;
    seq_length = input_seq_length;
    global_malloc_manage_float.set_head_zero();
    global_malloc_manage_int.set_head_zero();
    if(batchsize > max_train_batchsize){
        std::cout<<" batchsize: "<<batchsize<<" max_train_batchsize: "
                    <<max_train_batchsize<<std::endl;
        assert(batchsize <= max_train_batchsize);
    }
    if(seq_length > max_train_seq_length){
        std::cout<<" seq_length: "<<seq_length<<" max_train_seq_length: "
                    <<max_train_seq_length<<std::endl;
        assert(seq_length <= max_train_seq_length);
    }
}

