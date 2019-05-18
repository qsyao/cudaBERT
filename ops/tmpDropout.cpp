#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "cuDNN Error in File " << __FILE__ << 	 \
	 " Error on line " << __LINE__ << ": "      			 \
      << cudnnGetErrorString(status) << std::endl; 			 \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void printArr3D(float * arr, int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%f ", arr[i]);
        if(i % 10 == 9)
            printf("\n");
    }
    printf("\n");
}

void GPU_PrintArr3D(float* d_arr, int n)
{
    float* h_arr;
    h_arr = (float *) malloc(sizeof(float)*n);
    cudaMemcpy(h_arr,d_arr,sizeof(float)*n,cudaMemcpyDeviceToHost);
    printArr3D(h_arr,n);
}

class Dropout
{
public:
    cudnnHandle_t cudnn;
    cudnnDropoutDescriptor_t dropout_desc_;
    size_t states_size_in_bytes_;
    size_t reserve_space_size_in_bytes__;

    cudnnTensorDescriptor_t data_desc_;

    float dropRate;
    int n;
    float* ref_input{nullptr};
    float* d_dropout_out{nullptr};
    float* d_dx_dropout{nullptr};
    void* states_data;
    void* dropout_reserve_space;
    int batchSize, features, imgH, imgW;
    int in_out_bytes;

    Dropout(float dropR, int len) :
            dropRate(dropR), n(len)
    {
        in_out_bytes = sizeof(float)*n;

        checkCUDNN(cudnnCreate(&cudnn));
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc_));
        checkCUDNN(cudnnCreateTensorDescriptor(&data_desc_));

        std::cout << "states_size_in_bytes_: " << states_size_in_bytes_ << std::endl;
        checkCUDNN(cudnnDropoutGetStatesSize(cudnn,
                                             &states_size_in_bytes_));
        std::cout << "states_size_in_bytes_: " << states_size_in_bytes_ << std::endl;

        cudaMalloc(&states_data, states_size_in_bytes_);

        checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc_,
                                             cudnn,
                                             dropRate,
                                             states_data,
                                             states_size_in_bytes_,
                /*Seed*/1));
        std::cout << "PPPPPP" << std::endl;
    };

    float* Forward(float* d_input)
    {
        cudaMalloc(&d_dropout_out, in_out_bytes);
        cudaMalloc(&d_dx_dropout, in_out_bytes);
        ref_input = d_input;

        checkCUDNN(cudnnSetTensor4dDescriptor(data_desc_,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n,
                                              1,
                                              1,
                                              1));

        checkCUDNN(cudnnDropoutGetReserveSpaceSize(data_desc_,
                                                   &reserve_space_size_in_bytes__));
        cudaMalloc(&dropout_reserve_space,reserve_space_size_in_bytes__);

        printf("Input \n");
        GPU_PrintArr3D(d_input, n);
        checkCUDNN(cudnnDropoutForward(cudnn,
                                       dropout_desc_,
                                       data_desc_,
                                       ref_input,
                                       data_desc_,
                                       d_dropout_out,
                                       dropout_reserve_space,
                                       reserve_space_size_in_bytes__));

        printf("Dropout \n");
        GPU_PrintArr3D(d_dropout_out,n);

        return d_dropout_out;
    }

    float* Backward(float *d_in_grads)
    {
        checkCUDNN(cudnnDropoutBackward(cudnn,
                                        dropout_desc_,
                                        data_desc_,
                                        d_in_grads,
                                        data_desc_,
                                        d_dx_dropout,
                                        dropout_reserve_space,
                                        reserve_space_size_in_bytes__));

        printf("Dropout Grad\n");
        GPU_PrintArr3D(d_dx_dropout, n);
        return d_dx_dropout;
    }

};

int main()
{
    int batchSize = 1;
    int imgH = 5;
    int imgW = 5;
    int inC = 3;
    float dropRate = 0.1;

    int n = batchSize*imgH*imgW*inC;
    int in_out_bytes = n*sizeof(float);

    float *h_input, *d_input;
    float *d_output;
    float *h_grads, *d_grads;
    h_input = (float *) malloc(in_out_bytes);
    h_grads = (float *) malloc(in_out_bytes);
    cudaMalloc(&d_input,in_out_bytes);
    cudaMalloc(&d_grads,in_out_bytes);

    for(int i = 0; i < n; i++)
        h_input[i] = rand();

    Dropout dLayer = Dropout(dropRate, n);

    cudaMemcpy(d_input,h_input,in_out_bytes,cudaMemcpyHostToHost);

    d_output = dLayer.Forward(d_input);

    cudaMemcpy(h_grads,d_output, in_out_bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < batchSize*imgH*imgW*inC; i++)
        h_grads[i] *= 2;

    cudaMemcpy(d_grads,h_grads,in_out_bytes, cudaMemcpyHostToDevice);
    d_output = dLayer.Backward(d_grads);

}