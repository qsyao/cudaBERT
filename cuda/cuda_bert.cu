#include "cuda_runtime.h"
#include <time.h>

#include "utils/common.h"
#include "utils/inference.cu"
#include "utils/load_model.h"
#include "utils/manager.h"

int* filling_inputs(int* tensor, int seq_length, int start_length, int batchsize){
    int* target = (int*)malloc(sizeof(int)*seq_length*batchsize);
    for(int i = 0; i < seq_length-1 ; i++){
        target[i] = tensor[i%(start_length-1)];
    }
    target[seq_length-1] = tensor[start_length-1];
    for(int i=1 ; i<batchsize; i++){
        memcpy(target + seq_length*i, target, seq_length*sizeof(int));
    }
    return target;
}

extern "C"{

void get_gpu_result(global_manager * handle,
        float* gpu_tensor, 
        float* output, 
        int total_size) {
    checkCudaErrors(cudaMemcpyAsync(output, 
                   gpu_tensor, 
                   sizeof(float)*total_size, 
                   cudaMemcpyDeviceToHost,
                   handle->get_cal_stream()));
    
    cudaStreamSynchronize(handle->get_cal_stream());  
}

Retval BERT_Inference (global_manager * handle,
                    int* words, 
                    int* token_types, 
                    int batchsize, 
                    int seq_length, 
                    int* attention_mask);

global_manager * init_model(bool large = false, char dir[] = ""){
    global_manager * handle = new global_manager(large, dir);
    return handle;
}

void cuda_classify (global_manager * handle,
                    float* output,
                    int* words, 
                    int* token_types, 
                    int batchsize, 
                    int seq_length,
                    int num_classes,
                    int* attention_mask){
    Retval ret = BERT_Inference(
                            handle,
                            words,
                            token_types,
                            batchsize,
                            seq_length,
                            attention_mask);
    float * output_gpu;
    output_gpu = classify_inference(handle, ret.pooled_output, num_classes);
    get_gpu_result(handle, output_gpu, output, batchsize*num_classes);
}

void check_model(global_manager *handle){
    std::cout<<handle->dir_npy<<std::endl;
    std::cout<<handle->hidden_size<<std::endl;
    std::cout<<handle->num_hidden_layers<<std::endl;
    std::cout<<handle->intermediate_size<<std::endl;
}

void check_inputs(int *a, int n){
    for (int i=0; i<n; i++) {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
}

void test(int batchsize, int seq_length, int nIter, bool base){
    global_manager * handle = init_model(base);

    int test_word_id_seed[11] = {2040, 2001, 3958, 27227, 1029, 3958, 103,
                               2001, 1037, 13997, 11510};
    int test_token_type_id_seed[11] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

    int attention_mask[11] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};

    int *test_word_id, *test_token_type_id, *test_attention_mask;
    test_word_id = filling_inputs(test_word_id_seed, seq_length, 11, batchsize);
    test_token_type_id = filling_inputs(test_token_type_id_seed, seq_length, 11, batchsize);
    test_attention_mask = filling_inputs(attention_mask, seq_length, 11, batchsize);
    std::cout<<" Seq_length : "<<seq_length<<std::endl;
    std::cout<<" Batchsize : "<<batchsize<<std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm Up
    for(int i = 0; i < 10; i++){
        BERT_Inference(handle, 
                       test_word_id, 
                       test_token_type_id, 
                       batchsize, 
                       seq_length, 
                       test_attention_mask);
    }

    double total_time = 0;
    for(int i = 0; i < nIter; i++){
        float it_time;
        cudaEventRecord(start);
        BERT_Inference(handle, 
                       test_word_id, 
                       test_token_type_id, 
                       batchsize, 
                       seq_length, 
                       test_attention_mask);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&it_time, start, stop);
        total_time += it_time;
    }    

    delete handle;

    double dSeconds = total_time/(double)nIter;
    printf("Time= %.2f(ms)\n", dSeconds);
}
} // extern "C"

//int main(){
//    test();
//    return 0;
//}

//nvcc cuda_bert.cu -o test -lcublas -I /usr/local/cuda-9.0/samples/common/inc/ -lcnpy -L ./ --std=c++11
