#ifndef SOFTMAX2_CUDA_BERT
#define SOFTMAX2_CUDA_BERT

#include "op_kernel.cuh"

class op_SoftMax : public op_kernel {
public:
    op_SoftMax(global_handle *handle)
            : op_kernel(handle) {};

    ~op_SoftMax();

    template<typename T>
    void forward(
            T *tensor,
            size_t n1,
            size_t n2
    );

    template <typename T>
    void backward(T *dout, size_t n1, size_t n2);

    void update_weights();

private:
    size_t warpsize;

public:
    float *grad_input;
};

#endif
