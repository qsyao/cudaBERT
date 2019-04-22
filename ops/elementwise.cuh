#ifndef BERT_CUDA_ELEMENTWISE
#define BERT_CUDA_ELEMENTWISE

#include "op_kernel.cuh"

template <typename T>
void copy_pooler(T* &output, T* tensor, global_handle* handle);

class op_FusionTranspose : public op_kernel{
  public:
    op_FusionTranspose(global_handle* handle)
               : op_kernel(handle) {};

    ~op_FusionTranspose();

    template <typename T>
    void forward (T* &out,
                T* in, 
                int num, 
                bool muti_head);
    
    void backward() {}
};

class op_Gelu : public op_kernel {
  public:
    op_Gelu(global_handle* handle)
               : op_kernel(handle) {};

    ~op_Gelu();

    template <typename T>
    void forward (T* tensor, size_t max_num) ;

    template <typename T>
    void backward (T* tensor,  size_t max_num) {}
};

class op_Tanh : public op_kernel {
  public:
    op_Tanh(global_handle* handle)
            : op_kernel(handle) {};

    ~op_Tanh();

    template <typename T>
    void forward (T* tensor, size_t max_num) ;

    template <typename T>
    void backward (T* tensor, size_t max_num) {}
};

class op_Mask_Add : public op_kernel{
  /* 
    Fuse Div in Mask
  */
  public:
    op_Mask_Add(global_handle* handle)
        : op_kernel(handle) {};

    ~op_Mask_Add();

    template <typename T>
    void forward (
                  T* tensor, 
                  int* mask, 
                  float number);

    template <typename T>
    void backward (
                  T* tensor, 
                  int* mask, 
                  float number) {}

};

#endif
