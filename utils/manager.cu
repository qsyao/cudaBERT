#include "manager.cuh"
#include "load_model.h"

global_handle::global_handle (bool BERT_Large, std::string dir, bool optimRunningTime, bool isTrain, int numClasses) {
    if(BERT_Large){
        dir_npy = "model_npy/large_uncased";
        hidden_size = 1024;
        num_hidden_layers = 24;
        num_attention_heads = 16;
        intermediate_size = 4096;
        max_mem_size = 200 * 512;
    }
    if (dir != "") 
        dir_npy = dir;
    optim_running_time = optimRunningTime;
    is_train = isTrain;
    num_classes = numClasses;
    load_from_dir_to_GPU(dir_npy, tts);
    checkError(cublasCreate(&handle), "cublasCreate() error!\n");
    init_cudamemory(max_mem_size / max_seq_length, max_seq_length);
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
}

global_handle::~global_handle(){
    global_malloc_manage_float.del();
    global_malloc_manage_int.del();
    checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
}

void global_handle::init_cudamemory(int batchsize, int seq_length){
    global_malloc_manage_int.init(batchsize * seq_length * 4);

    size_t left, total, real_Memcost;
    checkCudaErrors(cudaMemGetInfo(&left, &total));

    left -= 1024 * 1024 * 500;

    std::cout<<"CUDA Memory INFO: Free: "<< left / 1024 / 1024 <<"MB"<<std::endl;
    left = left / sizeof(float);
    global_malloc_manage_float.init(left);
    
    while(1){
        //TODO: train or inference
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
        if (real_Memcost < left)
            break;
        else
            batchsize = batchsize * 9 / 10;
    }
    
    max_mem_size = batchsize * seq_length;
    std::cout<<"Support max_seq_length: "<<max_seq_length<<" max_batchsize: "
                <<batchsize<<" approximate max_size: "<<max_mem_size<<std::endl;
    
}

void global_handle::set_scale(size_t input_batchsize, size_t input_seq_length){
    batchsize = input_batchsize;
    seq_length = input_seq_length;
    if(batchsize * seq_length > max_mem_size){
        std::cout<<"Error : Batchsize * Seq_lengh is too big too alloc"<<std::endl;
        std::cout<<" batchsize: "<<batchsize<<" seq_length: "
                    <<seq_length<<" max_size: "<<max_mem_size<<std::endl;
        assert(batchsize * seq_length <= max_mem_size);
    }
}

