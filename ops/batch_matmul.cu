#include "batch_matmul.cuh"
#include "matmul.cuh"

template <typename T>
void op_Batch_Matmul::forward(size_t batchsize,
                            size_t n,
                            size_t k,
                            size_t m,
                            T* input_a,
                            T* input_b,
                            T* &output,
                            bool transpose_a,
                            bool transpose_b){
    output =  handle->global_malloc_manage_float.get_new_head_point(
                                                    batchsize * n * m);
    
    std::vector<size_t> a_shape, b_shape, output_shape;
    if(!transpose_a)
        a_shape = {batchsize, n, k};
    else
        a_shape = {batchsize, k, n};
    if(!transpose_b)
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
                                    float* input_a,
                                    float* input_b,
                                    float* &output,
                                    bool transpose_a,
                                    bool transpose_b);