#include "dropout.cuh"

template <typename T>
void op_Dropout::forward(T *&output, T *input, int length) {
    output = handle->global_malloc_manage_float.get_new_head_point(length);

    checkCUDNN(cudnnDropoutForward(cudnn,
                                   dropout_desc_,
                                   data_desc_,
                                   input,
                                   data_desc_,
                                   output,
                                   dropout_reserve_space,
                                   reserve_space_size_in_bytes_));
}

template
void op_Dropout::forward(float *&output, float *input, int length);

template<typename T>
void op_Dropout::backward(T *dout) {
    grad_input = handle->global_malloc_manage_float.get_new_head_point(n);
    checkCUDNN(cudnnDropoutBackward(cudnn,
                                    dropout_desc_,
                                    data_desc_,
                                    dout,
                                    data_desc_,
                                    grad_input,
                                    dropout_reserve_space,
                                    reserve_space_size_in_bytes_));
}

template
void op_Dropout::backward(float *dout);