#ifndef BERT_CUDA_BATCH_MATMUL
#define BERT_CUDA_BATCH_MATMUL

#include "op_kernel.cuh"

class op_Batch_Matmul : public op_kernel {
  public:
    op_Batch_Matmul(global_handle* handle)
                : op_kernel(handle) {};   

    template <typename T>
    void forward(size_t batchsize,
                 size_t n,
                 size_t k,
                 size_t m,
                 T* input_a,
                 T* input_b,
                 T* &output,
                 bool transpose_a,
                 bool transpose_b);

    void backward() {}
};

#endif