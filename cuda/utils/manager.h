#ifndef CUDA_BERT_MANAGER
#define CUDA_BERT_MANAGER

#include<string>
#include<vector>
#include "cuda_runtime.h"
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#include "common.h"
#include "load_model.h"

__global__ void MemoryCpyLinear(float* out, float* in, int max, int warpsize) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max; i += gridDim.x * blockDim.x)
        out[i] = in[i%warpsize];
    __syncthreads();
}

extern "C"
class global_manager {
    public:
        global_manager (bool BERT_Large=false, std::string dir = "") {
            if(BERT_Large){
                dir_npy = "model_npy/large_uncased";
                hidden_size = 1024;
                num_hidden_layers = 24;
                num_attention_heads = 16;
                intermediate_size = 4096;
                max_mem_size = 80 * 1000;
            }
            if (dir != "") 
                dir_npy = dir;
            load_from_dir_to_GPU(dir_npy, tts);
            weights = load_dict_weights(tts, num_hidden_layers);
            checkError(cublasCreate(&handle), "cublasCreate() error!\n");
            init_cudamemory(max_batchsize, max_mem_size / max_batchsize);
            prepare_linear(this, tts, weights);
            cudaStreamCreate(&cal_stream);
            cudaStreamCreate(&copy_stream);
            cudaEventCreate(&copy_event);
            checkError(cublasSetStream(handle, cal_stream), "Set cublas stream Error!\n");
        }

        ~global_manager(){
            global_malloc_manage_float.del();
            global_malloc_manage_int.del();
            checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
        }

        void init_cudamemory(int batchsize, int seq_length){
            global_malloc_manage_float.init(batchsize*seq_length*hidden_size + 
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
                                            batchsize*seq_length*hidden_size);

            global_malloc_manage_int.init(batchsize * seq_length * 4);
        }


        void prepare_linear(global_manager *handle,
                    std::vector<tagged_tensor *>& tts, 
                    dict_weights &weights) {
            size_t hidden_size = handle->hidden_size;
            size_t max_mem_size = handle->max_mem_size;

            for(int i = 0; i < handle->num_hidden_layers; i++){
                float *batch_attentin_weights, *key, *value;
                checkCudaErrors(cudaMalloc((void**)&batch_attentin_weights, 
                        sizeof(float) * hidden_size * hidden_size * 3));
                key = batch_attentin_weights + 1 * hidden_size * hidden_size;
                value = batch_attentin_weights + 2 * hidden_size * hidden_size;

                dim3 threads(512, 1, 1);
                dim3 blocks(hidden_size * hidden_size/512 + 1, 1, 1);
                MemoryCpyLinear<<<blocks, threads>>>(batch_attentin_weights, 
                                                      weights["attention_self_query_kernel"][i],
                                                      hidden_size * hidden_size,
                                                      hidden_size * hidden_size);
                MemoryCpyLinear<<<blocks, threads>>>(key, 
                                                      weights["attention_self_key_kernel"][i],
                                                      hidden_size * hidden_size,
                                                      hidden_size * hidden_size);
                MemoryCpyLinear<<<blocks, threads>>>(value, 
                                                      weights["attention_self_value_kernel"][i],
                                                      hidden_size * hidden_size,
                                                      hidden_size * hidden_size);

                checkCudaErrors(cudaFree(weights["attention_self_key_kernel"][i]));
                checkCudaErrors(cudaFree(weights["attention_self_query_kernel"][i]));
                checkCudaErrors(cudaFree(weights["attention_self_value_kernel"][i]));
                weights["attention_self_query_kernel"][i] = batch_attentin_weights;
                weights["attention_self_key_kernel"][i] = key;
                weights["attention_self_value_kernel"][i] = value;
            }
            /*
            for(auto iter = weights.begin(); iter != weights.end(); iter++){
                size_t find = iter->first.find("bias");
                if(find != std::string::npos){
                    tagged_tensor* tt;
                    std::vector<std::string> keys = {iter->first};
                    tt = look_up_tts(tts, keys);
                    size_t warpsize = tt->shape[0];
                    for(int i = 0; i < handle->num_hidden_layers; i++){
                        float *ret;
                        checkCudaErrors(cudaMalloc((void**)&ret, 
                                sizeof(float) * warpsize * max_mem_size));
                        dim3 threads(512, 1, 1);
                        dim3 blocks(warpsize * max_mem_size/512 + 1, 1, 1);
                        MemoryCpyLinear<<<blocks, threads>>>(ret,
                                                            iter->second[i],
                                                            max_mem_size * warpsize,
                                                            warpsize);
                        iter->second[i] = ret;
                        //debug_tensor_gpu<float>(std::string("Init bias"), ret, 11, warpsize, 100);
                    }                 
                }
            }
            */
        }

        void set_scale(size_t input_batchsize, size_t input_seq_length){
            batchsize = input_batchsize;
            seq_length = input_seq_length;
            if(batchsize * seq_length > max_mem_size){
                std::cout<<"Error : Batchsize * Seq_lengh is too big too alloc"<<std::endl;
                assert(batchsize * seq_length <= max_mem_size);
            }
        }

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
