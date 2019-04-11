#ifndef CUDA_BERT_COMMEN
#define CUDA_BERT_COMMEN

#include<iostream>
#include<string>
#include<vector>
#include<cstdlib>
#include<unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

// These are CUDA Helper functions (in addition to helper_cuda.h)
void inline checkError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}

typedef std::unordered_map<std::string, std::vector<float *>> dict_weights;

class tagged_tensor{
    public:
    
        tagged_tensor(std::string name, std::vector<size_t> shape,
                    float* cpu_mem)
                    : name(name), shape(shape), cpu_mem(cpu_mem) {
                        num_elements = get_num_elements(shape);
                    };
      
        size_t get_num_elements(std::vector<size_t> shape){
            size_t num = 1;
            for(int i = 0; i < shape.size(); i++){
                num *= shape[i];
            }
            return num;
        }

        std::string debug_string(){
            std::string out;
            std::string str_shape;
            for(auto t : shape){
                str_shape += "  " + std::to_string(t) + "  ";
            }
            out += " **** " + name + '\n';
            if(cpu_mem != nullptr)
                out += " The first float : " + std::to_string(*cpu_mem) + '\n';
            else out += "kernel on GPU \n";
            out += " The num_elements : " + std::to_string(num_elements) + '\n';
            out += " Shape : " + str_shape + '\n';
            return out;
        }
        
        std::string name;
        std::vector<size_t> shape;
        float* cpu_mem;
        float* gpu_mem;
        size_t num_elements;
};

template <typename T>
class malloc_manage {
  public:
    void init(long t_size) {
        tot_size = t_size;
        head = 0;
        checkCudaErrors(cudaMalloc((void **)&point_head, tot_size * sizeof(T)));
        return ;
    }
    T *get_new_head_point(long t_size = 0) {
        T* now = point_head + head;
        head += t_size;
        assert(head <= tot_size);
        return now;
    }

    void set_head_zero() {
        head = 0;
    }

    void record_layer_start(){
        layer_start = head;
    }
    
    void reuse_layer_mem(){
        head = layer_start;
    }
    void del() {
        checkCudaErrors(cudaFree(point_head));
        return ;
    }

  private:
    T *point_head;
    long layer_start;
    long tot_size;
    long head;

};

template <typename T>
void debug_tensor(std::string tag, T* tensor, int max_x , int length_x, int max_y = 1){
    /*
       print 1 to max_y lines from tensor:
       length_x: len(rows)
       max_x : len(rows) to print
    */
    std::cout<<" DEBUG TENSOR: ****  "<<tag<<"  *****  "<<std::endl;
    for(int i = 0; i < max_y; i++){
        for(int j = 0; j < max_x; j++)
            std::cout<<" "<<tensor[i*length_x + j]<<" ";
        std::cout<<std::endl;
    }
    std::cout<<" **************  DEBUG ENDING"<<std::endl;
}

template <typename T>
void debug_tensor_gpu(std::string tag, void* gpu_tensor, int max_x, int length_x, int max_y = 1){
    int length = length_x * max_y;
    T* cpu_mem;
    cpu_mem = (T *)malloc(sizeof(T) * length);
    T* tensor = static_cast<T *>(gpu_tensor);
    std::cout<<" \nDEBUG GPU TENSOR: ****  "<<tag<<"  ******  "<<std::endl;
    std::cout<<" Pointer:  --- "<<gpu_tensor<<" ------ "<<std::endl;
    checkCudaErrors(cudaMemcpyAsync(cpu_mem, tensor, sizeof(T)*length, cudaMemcpyDeviceToHost));
    for(int i = 0; i < max_y; i++){
        for(int j = 0; j < max_x; j++)
            std::cout<<" "<<cpu_mem[i*length_x + j]<<" ";
        std::cout<<" ... "<<cpu_mem[i*length_x + length_x -3]<<"  "
                          <<cpu_mem[i*length_x + length_x -2]<<"  "
                          <<cpu_mem[i*length_x + length_x -1]<<"  ";
        std::cout<<std::endl;
    }
    free(cpu_mem);
    std::cout<<" **************  DEBUG ENDING\n"<<std::endl;
}

template <typename T>
void dump_tensor(std::string file_name, 
                void* gpu_tensor,
                size_t dim1=0,
                size_t dim2=0,
                size_t dim3=0,
                size_t dim4=0,
                size_t dim5=0) {
    std::vector<size_t> shape;
    if(dim1 != 0) shape.push_back(dim1);
    if(dim2 != 0) shape.push_back(dim2);
    if(dim3 != 0) shape.push_back(dim3);
    if(dim4 != 0) shape.push_back(dim4);
    if(dim5 != 0) shape.push_back(dim5);
    long length = 1;
    for(auto dim : shape) length *= dim;
    T* cpu_mem;
    cpu_mem = (T *)malloc(sizeof(T) * length);
    T* tensor = static_cast<T *>(gpu_tensor);
    checkCudaErrors(cudaMemcpyAsync(cpu_mem, tensor, sizeof(T)*length, cudaMemcpyDeviceToHost));
    cnpy::npy_save("debug/cubert/" + file_name + ".npy", cpu_mem, shape);
    free(cpu_mem);
    return ;
}


extern "C"
struct Retval{
    float *tensor;
    float *pooled_output;
};

#endif
/*
attention_output_dense_bias   768
attention_output_dense_kernel 768, 768
attention_self_key_bias     768
attention_self_key_kernel   768, 768
attention_self_query_bias   768
attention_self_query_kernel 768, 768
attention_self_value_bias   768
attention_self_value_kernel 768, 768
intermediate_dense_bias     3072
intermediate_dense_kernel   768, 3072
output_dense_bias           768
output_dense_kernel         3072, 768
*/