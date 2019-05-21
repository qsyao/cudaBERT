#include "cuda_runtime.h"
#include <time.h>

#include "utils/common.h"
#include "utils/bert.cuh"

#include "cuda_bert.cuh"

#include <algorithm>
#include <vector>

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

bert* init_model(bool large = false, 
                 int num_gpu = 0, 
                 int max_batchsize = 0,
                 int max_seq_length = 0,
                 char dir[] = ""){
    bert* ret = new bert(large, num_gpu, dir, max_batchsize, max_seq_length);
    return ret;
}

Retval Cuda_Inference(bert* model,
                    int* words,
                    int* token_types,
                    int batchsize,
                    int seq_length,
                    int* masks){
    model->BERT_Inference(  words,
                            token_types,
                            batchsize,
                            seq_length,
                            masks);
    
    return model->ret;
}

void Cuda_Bert (bert* model,
                    float* output,
                    int* words,
                    int* token_types,
                    int batchsize,
                    int seq_length,
                    int* attention_mask){
    model->BERT_Inference(  words,
                            token_types,
                            batchsize,
                            seq_length,
                            attention_mask);

    model->get_gpu_result(output, model->ret.pooled_output, 
                batchsize * model->handle->hidden_size);
}

// void Cuda_Classify (bert* model,
//                     float* output,
//                     int* words,
//                     int* token_types,
//                     int batchsize,
//                     int seq_length,
//                     int num_classes,
//                     int* attention_mask){
//     model->BERT_Inference(  words,
//                             token_types,
//                             batchsize,
//                             seq_length,
//                             attention_mask);
//     float * output_gpu;
//     output_gpu = model->classify_inference(num_classes);

//     model->get_gpu_result(output, output_gpu, batchsize*num_classes);
// }

void test(int batchsize, int seq_length, int nIter, bool large, int num_gpu){
    bert* model = init_model(large, num_gpu, batchsize, seq_length);

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
    if (large) std::cout<<"Using large Model"<<std::endl;
    else std::cout<<"Using Base Model"<<std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* output_pinned;
    checkCudaErrors(cudaMallocHost((void **)&output_pinned,
                   (1024) * model->handle->hidden_size * sizeof(float)));

    //Warm Up
    for(int i = 0; i < 10; i++){
        model->BERT_Inference(
                            test_word_id, 
                            test_token_type_id, 
                            batchsize, 
                            seq_length, 
                            test_attention_mask);
        model->get_gpu_result(output_pinned,
                            model->ret.pooled_output, 
                            model->handle->batchsize * model->handle->hidden_size);

       if ( i == 0 ) {
           debug_tensor<float>(std::string("unit_test"),
                               output_pinned,
                               10,
                               model->handle->hidden_size,
                               max(model->handle->batchsize/10, (long)1));
       }
    }

    std::vector<float> latency(nIter);
    double total_time = 0;
    for(int i = 0; i < nIter; i++){
        float it_time;
        cudaEventRecord(start);
        // float * output;
        // cuda_classify(
        //         model,
        //         output,
        //         test_word_id,
        //         test_token_type_id,
        //         classes,
        //         batchsize,
        //         seq_length,
        //         2,
        //         test_attention_mask
        // );
       model->BERT_Inference(
                           test_word_id,
                           test_token_type_id,
                           batchsize,
                           seq_length,
                           test_attention_mask);

       model->get_gpu_result(output_pinned,
                           model->ret.pooled_output,
                           model->handle->batchsize * model->handle->hidden_size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&it_time, start, stop);
        latency[i] = it_time;
        total_time += it_time;
    }

    std::sort(latency.begin(), latency.end());
    std::cout<<" 99 Latency : "<<latency[(int)(nIter * 0.99)]<<"ms"<<std::endl;
    std::cout<<" Throughput: "<<nIter * 1000 / total_time<<" lines per second"<<std::endl;
    delete model;

    double dSeconds = total_time/(double)nIter;
    printf("Time= %.2f(ms)\n\n", dSeconds);
}
} // extern "C"
