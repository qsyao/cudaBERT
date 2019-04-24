#include "linear.cuh"
#include "matmul.cuh"

template <typename T>
__global__ void MemoryCpyLinear(T* out, T* in, size_t max, size_t warpsize, float mul) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < max; i += gridDim.x * blockDim.x)
        out[i] = static_cast<T>(in[i%warpsize] * mul);
    __syncthreads();
}

template
__global__ void MemoryCpyLinear<float>(float* out, float* in, size_t max, size_t warpsize, float mul);

op_BatchedLinear::op_BatchedLinear( std::string key_query_kernel, 
                                    std::string key_query_bias,
                                    std::string key_key_kernel, 
                                    std::string key_key_bias,
                                    std::string key_val_kernel, 
                                    std::string key_val_bias,
                                    global_handle* handle)
                                    : op_kernel(handle)  {
    std::vector<tagged_tensor *> tts = handle->tts;

    std::vector<std::string> keys = {key_query_kernel};
    tagged_tensor* tt = look_up_tts(tts, keys);
    query_kernel = tt->gpu_mem;
    keys = {key_query_bias};
    tt = look_up_tts(tts, keys);
    query_bias = tt->gpu_mem;

    keys = {key_key_kernel};
    tt = look_up_tts(tts, keys);
    key_kernel = tt->gpu_mem;
    keys = {key_key_bias};
    tt = look_up_tts(tts, keys);
    key_bias = tt->gpu_mem;

    keys = {key_val_kernel};
    tt = look_up_tts(tts, keys);
    val_kernel = tt->gpu_mem;
    keys = {key_val_bias};
    tt = look_up_tts(tts, keys);
    val_bias = tt->gpu_mem;

    size_t hidden_size = handle->hidden_size;

    float *key, *value;
    checkCudaErrors(cudaMalloc((void**)&batch_attentin_weights, 
            sizeof(float) * hidden_size * hidden_size * 3));
    key = batch_attentin_weights + 1 * hidden_size * hidden_size;
    value = batch_attentin_weights + 2 * hidden_size * hidden_size;

    dim3 threads(512, 1, 1);
    dim3 blocks(hidden_size * hidden_size/512 + 1, 1, 1);
    MemoryCpyLinear<<<blocks, threads>>>(batch_attentin_weights, 
                                            query_kernel,
                                            hidden_size * hidden_size,
                                            hidden_size * hidden_size);
    MemoryCpyLinear<<<blocks, threads>>>(key, 
                                            key_kernel,
                                            hidden_size * hidden_size,
                                            hidden_size * hidden_size);
    MemoryCpyLinear<<<blocks, threads>>>(value, 
                                            val_kernel,
                                            hidden_size * hidden_size,
                                            hidden_size * hidden_size);

    checkCudaErrors(cudaFree(query_kernel));
    checkCudaErrors(cudaFree(key_kernel));
    checkCudaErrors(cudaFree(val_kernel));
    query_kernel = batch_attentin_weights;
    key_kernel = key;
    val_kernel = value;
}

template <typename T> 
void op_Linear::forward (
                        T* &output, 
                        T* input, 
                        size_t n, 
                        size_t k, 
                        size_t m,
                        bool is_prepare,
                        bool debug) {
    stored_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    checkCudaErrors(cudaMemcpyAsync(stored_input, input, n * k * sizeof(float), cudaMemcpyDeviceToDevice));

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


template<typename T>
__global__ void MemoryCpyLinearTranpose(T *out, T *in, int n, int m, int max) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max; i += gridDim.x * blockDim.x) {
        out[(i % m) * n + i / m] = in[i];
    }
    __syncthreads();
}

template<typename T>
void op_Linear::backward(T *dout, size_t n,
                         size_t k,
                         size_t m) {
    T *kernel_tranpose;
    kernel_tranpose = handle->global_malloc_manage_float.get_new_head_point(k * m);
    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long) 65535, (k * m + 1023) / 1024), 1, 1);
    MemoryCpyLinearTranpose<T> << < blocks, threads, 0, handle->copy_stream >> >
                                                        (kernel_tranpose, kernel, k, m, k * m);

    cudaEventRecord(handle->copy_event, handle->copy_stream);
    cudaStreamWaitEvent(handle->cal_stream, handle->copy_event, 0);
//    debug_tensor_gpu<float>(std::string("kernel_tranpose"), kernel_tranpose, k, k, m);

    std::vector <size_t> a_shape = {n, m};
    std::vector <size_t> b_shape = {m, k};
    std::vector <size_t> c_shape = {n, k};

    grad_input = handle->global_malloc_manage_float.get_new_head_point(n * k);

    matmul(handle->handle,
           dout,
           a_shape,
           kernel_tranpose,
           b_shape,
           grad_input,
           c_shape,
           false,
           false,
           1.0f,
           0.0f);

    T *input_tranpose;
    input_tranpose = handle->global_malloc_manage_float.get_new_head_point(n * k);
    dim3 threads1(1024, 1, 1);
    dim3 blocks1(min((long) 65535, (k * n + 1023) / 1024), 1, 1);
    MemoryCpyLinearTranpose<T> << < blocks1, threads1, 0, handle->copy_stream >> >
                                                          (input_tranpose, stored_input, n, k, n * k);

    cudaEventRecord(handle->copy_event, handle->copy_stream);
    cudaStreamWaitEvent(handle->cal_stream, handle->copy_event, 0);

    a_shape = {k, n};
    b_shape = {n, m};
    c_shape = {k, m};

    grad_kernel = handle->global_malloc_manage_float.get_new_head_point(k * m);
    matmul(handle->handle,
           input_tranpose,
           a_shape,
           dout,
           b_shape,
           grad_kernel,
           c_shape,
           false,
           false,
           1.0f,
           0.0f);

    grad_bias = handle->global_malloc_manage_float.get_new_head_point(n * m);
    dim3 threads2(1024, 1, 1);
    dim3 blocks2(min((long) 65535, (n * m + 1023) / 1024), 1, 1);
    MemoryCpyLinear<T> << < blocks2, threads2, 0, handle->copy_stream >> > (grad_bias, dout, n * m, n * m, n);
}

template
void op_Linear::backward<float>(
        float *dout, size_t n,
        size_t k,
        size_t m);

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