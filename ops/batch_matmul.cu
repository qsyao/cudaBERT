#include "batch_matmul.cuh"
#include "matmul.cuh"

template<typename T>
void op_Batch_Matmul::forward(size_t batchsize,
                              size_t n,
                              size_t k,
                              size_t m,
                              T *input_a,
                              T *input_b,
                              T *&output,
                              bool transpose_a,
                              bool transpose_b) {
    stored_input = handle->global_malloc_manage_float.get_new_head_point(batchsize * n * k);
    checkCudaErrors(
            cudaMemcpyAsync(stored_input, input_a, batchsize * n * k * sizeof(float), cudaMemcpyDeviceToDevice));

    kernel = handle->global_malloc_manage_float.get_new_head_point(batchsize * k * m);
    checkCudaErrors(cudaMemcpyAsync(kernel, input_b, batchsize * k * m * sizeof(float), cudaMemcpyDeviceToDevice));

//    debug_tensor_gpu<float>(std::string("kernel"), kernel, 3, k*m, batchsize);

    output = handle->global_malloc_manage_float.get_new_head_point(
            batchsize * n * m);

    std::vector <size_t> a_shape, b_shape, output_shape;
    if (!transpose_a)
        a_shape = {batchsize, n, k};
    else
        a_shape = {batchsize, k, n};
    if (!transpose_b)
        b_shape = {batchsize, k, m};
    else
        b_shape = {batchsize, m, k};
    output_shape = {batchsize, n, m};

    matmul(handle->handle,
           input_a,
           a_shape,
           input_b,
           b_shape,
           output,
           output_shape,
           transpose_a,
           transpose_b);
}

template
void op_Batch_Matmul::forward<float>(size_t batchsize,
                                     size_t n,
                                     size_t k,
                                     size_t m,
                                     float *input_a,
                                     float *input_b,
                                     float *&output,
                                     bool transpose_a,
                                     bool transpose_b);

template<typename T>
__global__ void MemoryCpyLinearTranpose(T *out, T *in, int batch_size, int n, int m, int max) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max; i += gridDim.x * blockDim.x) {
        for (int j = 0; j < batch_size; j++)
            out[j * n * m + (i % m) * n + i / m] = in[j * n * m + i];
    }
    __syncthreads();
}

template<typename T>
void op_Batch_Matmul::backward(T *dout, size_t batchsize, size_t n, size_t k, size_t m, bool transpose_a,
                               bool transpose_b) {
    std::vector <size_t> a_shape, b_shape, c_shape;
    a_shape = {batchsize, n, m};
    if (!transpose_b)
        b_shape = {batchsize, k, m};
    else
        b_shape = {batchsize, m, k};
    if(!transpose_a)
        c_shape = {batchsize, n, k};
    else
        c_shape = {batchsize, k, n};

    grad_input = handle->global_malloc_manage_float.get_new_head_point(batchsize * n * k);

    matmul(handle->handle,
           dout,
           a_shape,
           kernel,
           b_shape,
           grad_input,
           c_shape,
           false,
           !transpose_b,
           1.0f,
           0.0f);
    if(!transpose_a)
        a_shape = {batchsize, n, k};
    else
        a_shape = {batchsize, k, n};
    b_shape = {batchsize, n, m};
    if (!transpose_b)
        c_shape = {batchsize, k, m};
    else
        c_shape = {batchsize, m, k};
    if(!transpose_b) {
        grad_kernel = handle->global_malloc_manage_float.get_new_head_point(batchsize * k * m);
        matmul(handle->handle,
               stored_input,
               a_shape,
               dout,
               b_shape,
               grad_kernel,
               c_shape,
               !transpose_a,
               false,
               1.0f,
               0.0f);
    }
    else {
        grad_kernel = handle->global_malloc_manage_float.get_new_head_point(batchsize * k * m);
        T* tmp_grad_kernel = handle->global_malloc_manage_float.get_new_head_point(batchsize * k * m);
        matmul(handle->handle,
               stored_input,
               a_shape,
               dout,
               b_shape,
               tmp_grad_kernel,
               c_shape,
               !transpose_a,
               false,
               1.0f,
               0.0f);

        dim3 threads1(1024, 1, 1);
        dim3 blocks1(min((long) 65535, (batchsize * k * m + 1023) / 1024), 1, 1);
        MemoryCpyLinearTranpose<T> << < blocks1, threads1, 0, handle->cal_stream >> >
                                                              (grad_kernel, tmp_grad_kernel, batchsize, k, m, m * k);
    }

}

template
void op_Batch_Matmul::backward(float *dout, size_t batchsize, size_t n, size_t k, size_t m, bool transpose_a,
                               bool transpose_b);