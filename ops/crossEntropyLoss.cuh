#ifndef BERT_CUDA_CROSSENTROPYLOSS
#define BERT_CUDA_CROSSENTROPYLOSS

#include "op_kernel.cuh"

class op_CrossEntropyLoss: public op_kernel {
public:
    op_CrossEntropyLoss(global_handle *handle,
                        float* key_weights = nullptr)
            : op_kernel(handle) {
        wieghts = key_weights;
    }

    ~op_CrossEntropyLoss();

    template<typename T, typename U>
    void forward(T* &output, T *input, U *classes,
                 size_t n1, size_t n2);

    template<typename T, typename U>
    void backward(T *dout,
                  size_t n1, size_t n2, U *classes);

    void update_weights();

private:
    float *wieghts;
public:
    float *grad_input;
};

#endif