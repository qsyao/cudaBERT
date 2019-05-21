#include "dropout.cuh"

template <typename T>
void op_Dropout::forward(T *&output, T *input, int len) {
    n = len;

    checkCUDNN(cudnnSetTensor4dDescriptor(data_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, 1, 1, 1));

    checkCUDNN(cudnnDropoutGetReserveSpaceSize(data_desc_,
                                               &reserve_space_size_in_bytes_));

    dropout_reserve_space = handle->global_malloc_manage_float.get_new_head_point(reserve_space_size_in_bytes_);

    output = handle->global_malloc_manage_float.get_new_head_point(n);

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
void op_Dropout::forward(float *&output, float *input, int len);

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