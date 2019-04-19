#ifndef SOFTMAX2_CUDA_BERT
#define SOFTMAX2_CUDA_BERT

#include "op_kernel.cuh"

class op_SoftMax : public op_kernel {
  public:
    op_SoftMax(global_handle* handle)
               : op_kernel(handle) {};

    ~op_SoftMax();
    
    template<typename T> 
    void forward(
                T* tensor,
                size_t n1,
                size_t n2
                );

    void backward();

    void update_weights();

  private:
    size_t warpsize;

};

#endif
