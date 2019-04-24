#include "batch_matmul.cuh"

void blas_sgemm_batch(  cublasHandle_t handle,
                        const bool TransA, const bool TransB,
                        int m, int n, int k,
                        const float alpha,
                        const float **Aarray, int lda,
                        const float **Barray, int ldb,
                        const float beta,
                        float **Carray, int ldc,
                        int batchCount) {
    checkCudaErrors(cublasSgemmBatched(handle,
                                    TransA ? CUBLAS_OP_T : CUBLAS_OP_N,
                                    TransB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    Aarray, lda,
                                    Barray, ldb,
                                    &beta,
                                    Carray, ldc,
                                    batchCount));                 
}

__global__ void load_pointer_vector_qk(const float* query,
                                        const float* key,
                                        float* out,
                                        const float** query_array,
                                        const float** key_array,
                                        float** out_array,
                                        size_t batchsize,
                                        size_t seq_length,
                                        size_t num_attention_heads,
                                        size_t length_per_heads){
    size_t id_batchsize = threadIdx.x;
    size_t id_heads = blockIdx.x;
    size_t idx = id_batchsize * seq_length * length_per_heads * num_attention_heads +
                                length_per_heads * id_heads;
    query_array[id_batchsize * num_attention_heads + id_heads] = query + idx;
    key_array[id_batchsize * num_attention_heads + id_heads] = key + idx;
    out_array[id_batchsize * num_attention_heads + id_heads] = out +
        id_batchsize * seq_length * seq_length * num_attention_heads + id_heads * seq_length;

    __syncthreads();
}



void Query_Key::forward(const float* query,
                        const float* key,
                        float* &output){
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t num_attention_heads = handle->num_attention_heads;
    size_t length_per_heads = handle->hidden_size / num_attention_heads;

    output =  handle->global_malloc_manage_float.get_new_head_point(
                        batchsize * num_attention_heads * seq_length * seq_length);
    
    dim3 threads(batchsize, 1, 1);
    dim3 blocks(num_attention_heads, 1, 1);
    load_pointer_vector_qk<<<blocks, threads, 0, handle->cal_stream>>>(query,
                                                                       key,
                                                                       output,
                                                                       query_array,
                                                                       key_array,
                                                                       out_array,
                                                                       batchsize,
                                                                       seq_length,
                                                                       num_attention_heads,
                                                                       length_per_heads);
    
    blas_sgemm_batch(handle->handle,
                    true, false,
                    seq_length, seq_length, length_per_heads,
                    1.0,
                    key_array, num_attention_heads * length_per_heads,
                    query_array, num_attention_heads * length_per_heads,
                    0.0,
                    out_array, num_attention_heads * seq_length,
                    num_attention_heads * batchsize);
    
}

__global__ void load_pointer_vector_pv(const float* prob,
                                        const float* value,
                                        float* out,
                                        const float** prob_array,
                                        const float** value_array,
                                        float** out_array,
                                        size_t batchsize,
                                        size_t seq_length,
                                        size_t num_attention_heads,
                                        size_t length_per_heads){
    size_t id_batchsize = threadIdx.x;
    size_t id_heads = blockIdx.x;
    size_t idx = id_batchsize * seq_length * length_per_heads * num_attention_heads +
                                length_per_heads * id_heads;
    prob_array[id_batchsize * num_attention_heads + id_heads] = prob +         
         id_batchsize * seq_length * seq_length * num_attention_heads + id_heads * seq_length;
    value_array[id_batchsize * num_attention_heads + id_heads] = value + idx;
    out_array[id_batchsize * num_attention_heads + id_heads] = out + idx;

    __syncthreads();
}

void Prob_Value::forward(const float* prob,
                        const float* value,
                        float* &output){
    size_t batchsize = handle->batchsize;
    size_t seq_length = handle->seq_length;
    size_t num_attention_heads = handle->num_attention_heads;
    size_t length_per_heads = handle->hidden_size / num_attention_heads;

    output =  handle->global_malloc_manage_float.get_new_head_point(
                        batchsize * num_attention_heads * seq_length * seq_length);
    
    dim3 threads(batchsize, 1, 1);
    dim3 blocks(num_attention_heads, 1, 1);
    load_pointer_vector_pv<<<blocks, threads, 0, handle->cal_stream>>>(prob,
                                                                       value,
                                                                       output,
                                                                       prob_array,
                                                                       value_array,
                                                                       out_array,
                                                                       batchsize,
                                                                       seq_length,
                                                                       num_attention_heads,
                                                                       length_per_heads);
    
    blas_sgemm_batch(handle->handle,
                    false, false,
                    length_per_heads, seq_length, seq_length,
                    1.0,
                    value_array, num_attention_heads * length_per_heads,
                    prob_array, num_attention_heads * seq_length,
                    0.0,
                    out_array, num_attention_heads * length_per_heads,
                    num_attention_heads * batchsize);
    
}