#include "manager.cuh"

global_manager::global_manager (bool BERT_Large, int num_gpu, std::string dir) {
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
            checkCudaErrors(cudaSetDevice(num_gpu));
        }

global_manager::~global_manager(){
            global_malloc_manage_float.del();
            global_malloc_manage_int.del();
            checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
        }

void global_manager::init_cudamemory(int batchsize, int seq_length){
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


void global_manager::prepare_linear(global_manager *handle,
                    std::vector<tagged_tensor *>& tts, 
                    dict_weights &weights) {
            size_t hidden_size = handle->hidden_size;

            for(int i = 0; i < handle->num_hidden_layers; i++){
                float *batch_attentin_weights, *key, *value;
                checkCudaErrors(cudaMalloc((void**)&batch_attentin_weights, 
                        sizeof(float) * hidden_size * hidden_size * 3));
                key = batch_attentin_weights + 1 * hidden_size * hidden_size;
                value = batch_attentin_weights + 2 * hidden_size * hidden_size;

                dim3 threads(512, 1, 1);
                dim3 blocks(hidden_size * hidden_size/512 + 1, 1, 1);
                MemoryCpyLinear<float><<<blocks, threads>>>(batch_attentin_weights, 
                                                      weights["attention_self_query_kernel"][i],
                                                      hidden_size * hidden_size,
                                                      hidden_size * hidden_size);
                MemoryCpyLinear<float><<<blocks, threads>>>(key, 
                                                      weights["attention_self_key_kernel"][i],
                                                      hidden_size * hidden_size,
                                                      hidden_size * hidden_size);
                MemoryCpyLinear<float><<<blocks, threads>>>(value, 
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

void global_manager::set_scale(size_t input_batchsize, size_t input_seq_length){
            batchsize = input_batchsize;
            seq_length = input_seq_length;
            if(batchsize * seq_length > max_mem_size){
                std::cout<<"Error : Batchsize * Seq_lengh is too big too alloc"<<std::endl;
                assert(batchsize * seq_length <= max_mem_size);
            }
        }

