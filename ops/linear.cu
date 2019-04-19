#include "linear.cuh"
#include "elementwise.cuh"
#include "matmul.cuh"

template <typename T> 
void op_Linear::forward (
                        T* &output, 
                        T* input, 
                        size_t n, 
                        size_t k, 
                        size_t m,
                        bool is_prepare,
                        bool debug) {
    output = handle->global_malloc_manage_float.get_new_head_point(n * m);

    if (debug) {
        debug_tensor_gpu<T>(std::string("weights"), kernel, 2, 2, 10);
        debug_tensor_gpu<T>(std::string("bias"), bias, 2, 2, 1);
        debug_tensor_gpu<T>(std::string("input_Linear"), bias, 10, k, min((int)n,10));
    }

    if(!is_prepare){
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long)65535, n*m/1024) + 1, 1, 1);
        MemoryCpyLinear<T><<<blocks, threads, 0, handle->copy_stream>>>(
                                                 output, bias, n*m, m);
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(output,
                                    bias, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handle->copy_stream));
    }
    
    cudaEventRecord(handle->copy_event, handle->copy_stream);

    cudaStreamWaitEvent(handle->cal_stream, handle->copy_event, 0);

    if(debug)
        debug_tensor_gpu<T>(std::string("After Linear copy"), output,10, m, min((int)n,10));

    std::vector<size_t> a_shape={n, k};
    std::vector<size_t> b_shape={k, m};
    std::vector<size_t> c_shape={n, m};

    matmul(handle->handle, 
           input, 
           a_shape, 
           kernel, 
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

template 
void op_Linear::forward<float>(
                            float* &output, 
                            float* input, 
                            size_t n, 
                            size_t k, 
                            size_t m,
                            bool is_prepare,
                            bool debug);

template <typename T>
void op_BatchedLinear::forward(
                            T* &output, 
                            T* input, 
                            size_t n, 
                            size_t k, 
                            size_t m,
                            bool is_prepare,
                            bool debug) {
    output = handle->global_malloc_manage_float.get_new_head_point(3 * n * m);
    
    //dim3 threads(512, 1, 1);
    //dim3 blocks(max(3*n*m, 3*k*m)/512 + 1, 1, 1);
    //BatchMemoryCpyLinear<T><<<blocks, threads>>>(weights, output, weights_0, beta_0, weights_1,
    //                            beta_1, weights_2, beta_2, n, k, m);
    if(!is_prepare){
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long)65535, n*m/1024) + 1, 1, 1);
        MemoryCpyLinear<T><<<blocks, threads, 0, handle->copy_stream>>>(
                                                      output, query_bias, n*m, m);
        MemoryCpyLinear<T><<<blocks, threads, 0, handle->copy_stream>>>(
                                                 output + n*m, key_bias, n*m, m);
        MemoryCpyLinear<T><<<blocks, threads, 0, handle->copy_stream>>>(
                                              output + 2*n*m, val_bias, n*m, m);
    }
    else{
        checkCudaErrors(cudaMemcpyAsync(output,
                                    query_bias, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handle->copy_stream));
        checkCudaErrors(cudaMemcpyAsync(output + n*m, 
                                    key_bias, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handle->copy_stream));
        checkCudaErrors(cudaMemcpyAsync(output + 2*n*m, 
                                    val_bias, 
                                    n*m*sizeof(float), 
                                    cudaMemcpyDeviceToDevice,
                                    handle->copy_stream));
    }

    //dim3 threads2(512, 1, 1);
    //dim3 blocks2(k*m/512 + 1, 1, 1);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights, weights_0, k*m, k*m);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights + k*m, weights_1, k*m, k*m);
    //MemoryCpyLinear<T><<<blocks2, threads2>>>(weights + 2*k*m, weights_2, k*m, k*m);
    cudaEventRecord(handle->copy_event, handle->copy_stream);
    cudaStreamWaitEvent(handle->cal_stream, handle->copy_event, 0);

    if(debug){
        //debug_tensor_gpu<T>(std::string("inputs"), input, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("key"), weights_0, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("query"), weights_0+k*m, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("value"), weights_0+2*k*m, 10, 768, 11);
        debug_tensor_gpu<T>(std::string("before matmul"), output, 5, handle->hidden_size*handle->seq_length, handle->batchsize*3);
        //debug_tensor_gpu<T>(std::string("bias"), beta_0, 10, handle->hidden_size, 11);
    }
    
    std::vector<size_t> a_shape={3, n, k};
    std::vector<size_t> b_shape={3, k, m};
    std::vector<size_t> c_shape={3, n, m};

    matmul(handle->handle, 
           input, 
           a_shape, 
           batch_attentin_weights, 
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

template
void op_BatchedLinear::forward<float>(
                                    float* &output, 
                                    float* input, 
                                    size_t n, 
                                    size_t k, 
                                    size_t m,
                                    bool is_prepare,
                                    bool debug);