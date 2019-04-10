#ifndef LINEAR_CUDA_BERT
#define LINEAR_CUDA_BERT

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "matmul.cu"
#include "../utils/manager.h"

template <typename T>
__global__ void MemoryCpyLinear(T* out, T* in, int max, int warpsize) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max; i += gridDim.x * blockDim.x)
        out[i] = in[i%warpsize];
    __syncthreads();
}

template <typename T>
__global__ void BatchMemoryCpyLinear (
        T* weights_out, 
        T* beta_out, 
        T* weights_0, 
        T* beta_0,
        T* weights_1, 
        T* beta_1, 
        T* weights_2, 
        T* beta_2, 
        size_t n, 
        size_t k, 
        size_t m) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < 3 * n * m; i += gridDim.x * blockDim.x) {
        size_t num_beta = i / (n * m);
        switch(num_beta){
            case 0 : {beta_out[i] = beta_0[i % m]; break;}
            case 1 : {beta_out[i] = beta_1[i % m]; break;}
            case 2 : {beta_out[i] = beta_2[i % m]; break;}
        }
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < 3 * k * m; i += gridDim.x * blockDim.x) {
        size_t num_weights = i / (k * m);
        switch (num_weights) {
            case 0 : {weights_out[i] = weights_0[i % (k * m)]; break;}
            case 1 : {weights_out[i] = weights_1[i % (k * m)]; break;}
            case 2 : {weights_out[i] = weights_2[i % (k * m)]; break;}
        }
    }
    __syncthreads();
}

template <typename T> 
void Linear (global_manager *handler, 
        T* output, 
        T* input, 
        T* weights, 
        T* beta,
        size_t n, 
        size_t k, 
        size_t m,
        bool is_prepare = false,
        bool debug=false) {

    if (debug) {
        debug_tensor_gpu<T>(std::string("weights"), weights, 2, 2, 10);
        debug_tensor_gpu<T>(std::string("bias"), beta, 2, 2, 1);
        debug_tensor_gpu<T>(std::string("input_Linear"), input, 10, k, min((int)n,10));
    }

    if(!is_prepare){
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long)65535, n*m/1024) + 1, 1, 1);
        MemoryCpyLinear<T><<<blocks, threads, 0, handler->get_copy_stream()>>>(
                                                 output, beta, n*m, m);
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(output,
                                    beta, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handler->get_copy_stream()));
    }
    
    cudaEventRecord(handler->copy_event, handler->get_copy_stream());

    cudaStreamWaitEvent(handler->get_cal_stream(), handler->copy_event, 0);

    if(debug)
        debug_tensor_gpu<T>(std::string("After Linear copy"), output,10, m, min((int)n,10));

    std::vector<size_t> a_shape={n, k};
    std::vector<size_t> b_shape={k, m};
    std::vector<size_t> c_shape={n, m};

    matmul(handler->handle, 
           input, 
           a_shape, 
           weights, 
           b_shape, 
           output, 
           c_shape,
           false, 
           false, 
           1.0f, 
           1.0f);
    if(debug)
        debug_tensor_gpu<T>(std::string("Linear out"), output, 10, m, min((int)n,10));
}

template <typename T>
void Batch_Linear (global_manager *handler, 
                   T* output, 
                   T* input, 
                   T* weights_0, 
                   T* beta_0,
                   T* weights_1, 
                   T* beta_1, 
                   T* weights_2, 
                   T* beta_2, 
                   size_t n, 
                   size_t k, 
                   size_t m,
                   bool is_prepare = false,
                   bool debug = false) {
    T* weights = handler->global_malloc_manage_float.get_new_head_point(3 * k * m);
    
    //dim3 threads(512, 1, 1);
    //dim3 blocks(max(3*n*m, 3*k*m)/512 + 1, 1, 1);
    //BatchMemoryCpyLinear<T><<<blocks, threads>>>(weights, output, weights_0, beta_0, weights_1,
    //                            beta_1, weights_2, beta_2, n, k, m);
    if(!is_prepare){
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long)65535, n*m/1024) + 1, 1, 1);
        MemoryCpyLinear<T><<<blocks, threads, 0, handler->get_copy_stream()>>>(
                                                      output, beta_0, n*m, m);
        MemoryCpyLinear<T><<<blocks, threads, 0, handler->get_copy_stream()>>>(
                                                 output + n*m, beta_1, n*m, m);
        MemoryCpyLinear<T><<<blocks, threads, 0, handler->get_copy_stream()>>>(
                                              output + 2*n*m, beta_2, n*m, m);
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(output,
                                    beta_0, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handler->get_copy_stream()));
        checkCudaErrors(cudaMemcpyAsync(output + n*m, 
                                    beta_1, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handler->get_copy_stream()));
        checkCudaErrors(cudaMemcpyAsync(output + 2*n*m, 
                                    beta_2, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handler->get_copy_stream()));
    }

    dim3 threads2(512, 1, 1);
    dim3 blocks2(k*m/512 + 1, 1, 1);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights, weights_0, k*m, k*m);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights + k*m, weights_1, k*m, k*m);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights + 2*k*m, weights_2, k*m, k*m);
    cudaEventRecord(handler->copy_event, handler->get_copy_stream());
    cudaStreamWaitEvent(handler->get_cal_stream(), handler->copy_event, 0);

    if(debug){
        //debug_tensor_gpu<T>(std::string("inputs"), input, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("key"), weights_0, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("query"), weights_0+k*m, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("value"), weights_0+2*k*m, 10, 768, 11);
        debug_tensor_gpu<T>(std::string("before matmul"), output, 5, handler->hidden_size*handler->seq_length, handler->batchsize*3);
        //debug_tensor_gpu<T>(std::string("bias"), beta_0, 10, handler->hidden_size, 11);
    }
    
    std::vector<size_t> a_shape={3, n, k};
    std::vector<size_t> b_shape={3, k, m};
    std::vector<size_t> c_shape={3, n, m};

    matmul(handler->handle, 
           input, 
           a_shape, 
           weights_0, 
           b_shape, 
           output, 
           c_shape,
           false, 
           false, 
           1.0f, 
           1.0f, 
           0);
    //if(debug)
        //debug_tensor_gpu<T>(std::string("Linear out"), output, 11, 768, 11*3);
}

#endif
