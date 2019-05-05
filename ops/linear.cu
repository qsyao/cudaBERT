#include "linear.cuh"
#include "matmul.cuh"

template<typename T>
__global__ void MemoryCpyLinear(T *out, T *in, size_t max, size_t warpsize, float mul) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max; i += gridDim.x * blockDim.x)
        out[i] = static_cast<T>(in[i % warpsize] * mul);
    __syncthreads();
}

template
__global__ void MemoryCpyLinear<float>(float *out, float *in, size_t max, size_t warpsize, float mul);

op_BatchedLinear::op_BatchedLinear(std::string key_query_kernel,
                                   std::string key_query_bias,
                                   std::string key_key_kernel,
                                   std::string key_key_bias,
                                   std::string key_val_kernel,
                                   std::string key_val_bias,
                                   global_handle *handle)
        : op_kernel(handle) {
    std::vector < tagged_tensor * > tts = handle->tts;

    std::vector <std::string> keys = {key_query_kernel};
    tagged_tensor *tt = look_up_tts(tts, keys);
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
    checkCudaErrors(cudaMalloc((void **) &batch_attentin_weights,
                               sizeof(float) * hidden_size * hidden_size * 3));
    key = batch_attentin_weights + 1 * hidden_size * hidden_size;
    value = batch_attentin_weights + 2 * hidden_size * hidden_size;

    dim3 threads(512, 1, 1);
    dim3 blocks(hidden_size * hidden_size / 512 + 1, 1, 1);
    MemoryCpyLinear << < blocks, threads >> > (batch_attentin_weights,
            query_kernel,
            hidden_size * hidden_size,
            hidden_size * hidden_size);
    MemoryCpyLinear << < blocks, threads >> > (key,
            key_kernel,
            hidden_size * hidden_size,
            hidden_size * hidden_size);
    MemoryCpyLinear << < blocks, threads >> > (value,
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

template<typename T>
void op_Linear::forward(
        T *&output,
        T *input,
        size_t n,
        size_t k,
        size_t m,
        bool is_prepare,
        bool debug) {
    if(handle->is_train) {
        stored_input = input;
    }

    output = handle->global_malloc_manage_float.get_new_head_point(n * m);

    if (debug) {
        debug_tensor_gpu<T>(std::string("weights"), kernel, 768, 768, 2);
//        debug_tensor_gpu<T>(std::string("bias"), bias, 2, 2, 1);
        // debug_tensor_gpu<T>(std::string("input_Linear"), bias, 10, k, min((int)n,10));
    }

    if (!is_prepare) {
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long) 65535, n * m / 1024) + 1, 1, 1);
        MemoryCpyLinear<T> << < blocks, threads, 0, handle->copy_stream >> > (
                output, bias, n * m, m);
    } else {
        checkCudaErrors(cudaMemcpyAsync(output,
                                        bias,
                                        n * m * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        handle->copy_stream));
    }

    cudaEventRecord(handle->copy_event, handle->copy_stream);

    cudaStreamWaitEvent(handle->cal_stream, handle->copy_event, 0);

    if (debug)
        debug_tensor_gpu<T>(std::string("After Linear copy"), output, min(10, (int) m), m, min((int) n, 10));

    std::vector <size_t> a_shape = {n, k};
    std::vector <size_t> b_shape = {k, m};
    std::vector <size_t> c_shape = {n, m};

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
//    if(debug)
//        debug_tensor_gpu<T>(std::string("Linear out"), output, 10, m, min((int)n,10));
}

template
void op_Linear::forward<float>(
        float *&output,
        float *input,
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
__global__ void cuApplyLinearGradientBias(T *__restrict__ dout, T *grad_bias, const size_t n1, const size_t n2) {
    for (int i2 = threadIdx.x; i2 < n2; i2 += blockDim.x) {
        grad_bias[i2] = 0;
        for (int i1 = 0; i1 < n1; i1++)
            grad_bias[i2] += dout[i1 * n2 + i2];
    }
    __syncthreads();
}

template<typename T>
void
linearBackward(T *dout, T *kernel, T *stored_input, T *grad_input, T *grad_kernel, T *grad_bias, size_t n, size_t k,
               size_t m, global_handle *handle) {
    std::vector <size_t> a_shape = {n, m};
    std::vector <size_t> b_shape = {k, m};
    std::vector <size_t> c_shape = {n, k};

    matmul(handle->handle,
           dout,
           a_shape,
           kernel,
           b_shape,
           grad_input,
           c_shape,
           false,
           true,
           1.0f,
           0.0f);

    a_shape = {n, k};
    b_shape = {n, m};
    c_shape = {k, m};

    matmul(handle->handle,
           stored_input,
           a_shape,
           dout,
           b_shape,
           grad_kernel,
           c_shape,
           true,
           false,
           1.0f,
           0.0f);

    dim3 threads2(min((long) 1024, m), 1, 1);
    dim3 blocks2(1, 1, 1);
    cuApplyLinearGradientBias << < blocks2, threads2, 0, handle->cal_stream >> > (
            dout,
                    grad_bias,
                    n,
                    m);
}


void op_Linear::update_weights(size_t n1, size_t n2) {
    if (handle->optim_method == "sgd") {
        apply_sgd_running_time(kernel, grad_kernel, n1, handle);
        apply_sgd_running_time(bias, grad_bias, n2, handle);
    }
}

template<typename T>
void op_Linear::backward(T *dout, size_t n,
                         size_t k,
                         size_t m) {
    grad_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    grad_kernel = handle->global_malloc_manage_float.get_new_head_point(k * m);
    grad_bias = handle->global_malloc_manage_float.get_new_head_point(m);
    linearBackward(dout, kernel, stored_input, grad_input, grad_kernel, grad_bias, n, k, m, handle);

    if (handle->optim_running_time)
        update_weights(k * m, m);
}

template
void op_Linear::backward<float>(
        float *dout, size_t n,
        size_t k,
        size_t m);

template<typename T>
void op_BatchedLinear::forward(
        T *&output,
        T *input,
        size_t n,
        size_t k,
        size_t m,
        bool is_prepare,
        bool debug) {
    output = handle->global_malloc_manage_float.get_new_head_point(3 * n * m);
    if(handle->is_train) {
        stored_input = input;
    }

    //dim3 threads(512, 1, 1);
    //dim3 blocks(max(3*n*m, 3*k*m)/512 + 1, 1, 1);
    //BatchMemoryCpyLinear<T><<<blocks, threads>>>(weights, output, weights_0, beta_0, weights_1,
    //                            beta_1, weights_2, beta_2, n, k, m);
    if (!is_prepare) {
        dim3 threads(1024, 1, 1);
        dim3 blocks(min((long) 65535, n * m / 1024) + 1, 1, 1);
        MemoryCpyLinear<T> << < blocks, threads, 0, handle->copy_stream >> > (
                output, query_bias, n * m, m);
        MemoryCpyLinear<T> << < blocks, threads, 0, handle->copy_stream >> > (
                output + n * m, key_bias, n * m, m);
        MemoryCpyLinear<T> << < blocks, threads, 0, handle->copy_stream >> > (
                output + 2 * n * m, val_bias, n * m, m);
    } else {
        checkCudaErrors(cudaMemcpyAsync(output,
                                        query_bias,
                                        n * m * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        handle->copy_stream));
        checkCudaErrors(cudaMemcpyAsync(output + n * m,
                                        key_bias,
                                        n * m * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        handle->copy_stream));
        checkCudaErrors(cudaMemcpyAsync(output + 2 * n * m,
                                        val_bias,
                                        n * m * sizeof(float),
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

    if (debug) {
        //debug_tensor_gpu<T>(std::string("inputs"), input, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("key"), weights_0, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("query"), weights_0+k*m, 10, 768, 11);
        //debug_tensor_gpu<T>(std::string("value"), weights_0+2*k*m, 10, 768, 11);
        debug_tensor_gpu<T>(std::string("before matmul"), output, 5, handle->hidden_size * handle->seq_length,
                            handle->batchsize * 3);
        //debug_tensor_gpu<T>(std::string("bias"), beta_0, 10, handle->hidden_size, 11);
    }

    std::vector <size_t> a_shape = {3, n, k};
    std::vector <size_t> b_shape = {3, k, m};
    std::vector <size_t> c_shape = {3, n, m};

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

template<typename T>
__global__ void cuApplyBatchLinearGradInput(T *out, T *in1, T *in2, T *in3, T *in4, size_t max) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max; i += gridDim.x * blockDim.x) {
        out[i] = in1[i] + in2[i] + in3[i] + in4[i];
    }
    __syncthreads();
}

template
void op_BatchedLinear::forward<float>(
        float *&output,
        float *input,
        size_t n,
        size_t k,
        size_t m,
        bool is_prepare,
        bool debug);

void op_BatchedLinear::update_weights(size_t n1, size_t n2) {
    if (handle->optim_method == "sgd") {
        apply_sgd_running_time(query_kernel, grad_query_kernel, n1, handle);
        apply_sgd_running_time(key_kernel, grad_key_kernel, n1, handle);
        apply_sgd_running_time(val_kernel, grad_val_kernel, n1, handle);

        apply_sgd_running_time(query_bias, grad_query_bias, n2, handle);
        apply_sgd_running_time(key_bias, grad_key_bias, n2, handle);
        apply_sgd_running_time(val_bias, grad_val_bias, n2, handle);
    }
}

template<typename T>
void op_BatchedLinear::backward(T *dout, T *grad_short_cut, size_t n, size_t k, size_t m) {
    grad_query_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    grad_query_kernel = handle->global_malloc_manage_float.get_new_head_point(k * m);
    grad_query_bias = handle->global_malloc_manage_float.get_new_head_point(m);
    linearBackward(dout, query_kernel, stored_input, grad_query_input, grad_query_kernel, grad_query_bias, n, k, m,
                   handle);

    grad_key_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    grad_key_kernel = handle->global_malloc_manage_float.get_new_head_point(k * m);
    grad_key_bias = handle->global_malloc_manage_float.get_new_head_point(m);
    linearBackward(dout + n * m, key_kernel, stored_input, grad_key_input, grad_key_kernel, grad_key_bias, n, k, m,
                   handle);

    grad_val_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    grad_val_kernel = handle->global_malloc_manage_float.get_new_head_point(k * m);
    grad_val_bias = handle->global_malloc_manage_float.get_new_head_point(m);
    linearBackward(dout + 2 * n * m, val_kernel, stored_input, grad_val_input, grad_val_kernel, grad_val_bias, n, k,
                   m, handle);

    dim3 threads(1024, 1, 1);
    dim3 blocks(min((long) 65535, n * k / 1024) + 1, 1, 1);
    grad_input = handle->global_malloc_manage_float.get_new_head_point(n * k);
    cuApplyBatchLinearGradInput<T> << < blocks, threads, 0, handle->cal_stream >> > (
            grad_input, grad_query_input, grad_key_input, grad_val_input, grad_short_cut, n * k);

    if (handle->optim_running_time)
        update_weights(k * m, m);
}

template
void op_BatchedLinear::backward<float>(float *dout, float *grad_short_cut, size_t n, size_t k, size_t m);
