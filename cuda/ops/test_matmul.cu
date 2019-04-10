#ifndef MATMUL_CUDA_BERT
#define MATMUL_CUDA_BERT

#include <assert.h>
#include <time.h>
#include <iostream>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <assert.h>

void inline checkError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {

        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}

void deviceMatmul(cublasHandle_t handle, float *d_A, std::vector<size_t> a_shape, float *d_B,
                std::vector<size_t> b_shape, float *d_C, bool transpose_a_ = false, 
                bool transpose_b_ = false, const float alpha = 1.0f, const float beta = 0.0f,
                const int batchCount = -1, const long long int strideA = 0, 
                const long long int strideB = 0, const long long int strideC = 0) 
{
    size_t m, k, n;
    if (batchCount != -1)
    {
        std::pair<int, int> trans;
        trans.first = transpose_a_ ? 1 : 2;
        int a_dim_remaining = 3 - trans.first;
        m = a_shape[a_dim_remaining];
        k = a_shape[3 - a_dim_remaining];
        if(b_shape.size() ==2) 
        {
            trans.second = transpose_b_ ? 1 : 0;
            int b_dim_remaining = 1 - trans.second;
            n = b_shape[b_dim_remaining];
            assert( b_shape[1 - b_dim_remaining] == k);
        }
        else 
        {
            trans.second = transpose_b_ ? 2 : 1;
            int b_dim_remaining = 3 - trans.second;
            n = b_shape[b_dim_remaining];
            assert( b_shape[3 - b_dim_remaining] == k);
        } 
    }
    else
    {
        std::pair<int, int> trans;
        trans.first = transpose_a_ ? 0 : 1;
        trans.second = transpose_b_ ? 1 : 0;
        int a_dim_remaining = 1 - trans.first;
        int b_dim_remaining = 1 - trans.second;

        m = a_shape[a_dim_remaining];
        k = a_shape[1 - a_dim_remaining];
        n = b_shape[b_dim_remaining];
        assert( b_shape[1 - b_dim_remaining] == k);
    }

    cublasOperation_t blas_transpose_a = (transpose_a_ ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t blas_transpose_b = (transpose_b_ ? CUBLAS_OP_T : CUBLAS_OP_N);
    //std::cout<<"n "<<n<<", "<<"m "<<m<<", "<<"k "<<k<<std::endl;
    if (batchCount == -1)
    {
        cublasStatus_t ret =
            cublasSgemm(handle, blas_transpose_b, blas_transpose_a,
                        n, m, k, &alpha, d_B, transpose_b_ ? k : n, d_A,
                        transpose_a_ ? m : k, &beta, d_C, n);
        checkError(ret, "cublas Sgemm returned an error!\n");
    }
    else
    {
        cublasStatus_t ret =
            cublasSgemmStridedBatched(handle, blas_transpose_b, blas_transpose_a,
                                    n, m, k, &alpha, d_B, transpose_b_ ? k : n, 
                                    strideB, d_A, transpose_a_ ? m : k, strideA, 
                                    &beta, d_C, n, strideC, batchCount);
        checkError(ret, "cublas SgemmBatched returned an error!\n");
    }
    return ;
}

size_t inline cal_shape(std::vector<size_t> shape)
{
    size_t num = 1;
    for (int i = 0; i < shape.size(); i++)
    {
        num *= shape[i];
    }
    return num;
}

void matmul(cublasHandle_t handle, float *d_A, std::vector<size_t> a_shape, float *d_B,
                std::vector<size_t> b_shape, float *d_C, std::vector<size_t> c_shape, 
                bool transpose_a_ = false, bool transpose_b_ = false, const float alpha = 1.0f, 
                const float beta = 0.0f)
{
    if(a_shape.size() == 2 && b_shape.size() == 2) 
    {
        deviceMatmul(handle, d_A, a_shape, d_B, b_shape, d_C, transpose_a_, transpose_b_, alpha, beta);
    }
    else if(a_shape.size() == 3 && b_shape.size() == 2)
    {
        int batchCount = a_shape[0];
        assert(a_shape[0] == c_shape[0]);
        long long int strideA = (long long int)(cal_shape(a_shape) / batchCount);
        long long int strideB = 0;
        long long int strideC = (long long int)(cal_shape(c_shape) / batchCount);
        deviceMatmul(handle, d_A, a_shape, d_B, b_shape, d_C, transpose_a_, transpose_b_, alpha, beta, 
                    batchCount, strideA, strideB, strideC);
    }
    else
    {
        assert(a_shape.size() == 3);
        assert(b_shape.size() == 3);
        assert(c_shape.size() == 3);
        int batchCount = a_shape[0];
        assert(b_shape[0] == c_shape[0]);
        assert(a_shape[0] == c_shape[0]);

        long long int strideA = (long long int)(cal_shape(a_shape) / batchCount);
        long long int strideB = (long long int)(cal_shape(b_shape) / batchCount);
        long long int strideC = (long long int)(cal_shape(c_shape) / batchCount);
        deviceMatmul(handle, d_A, a_shape, d_B, b_shape, d_C, transpose_a_, transpose_b_, alpha, beta, 
                    batchCount, strideA, strideB, strideC);
    }
    return ;
}

void randInit(float *data, int size)
{
    for (int i = 0; i < size; ++i) 
    {
        data[i] = 0.0;
        // data[i] = 1.1;
    }
}

#endif

void test(int argv[])
{
    cublasHandle_t handle;
    checkError(cublasCreate(&handle), "cublasCreate() error!\n");

    std::cout<<argv[0]<<"  "<<argv[1]<<"  "<<argv[2]<<"  "<<argv[3]<<"  "<<std::endl;

    int batchCount = argv[0];
    std::vector<size_t> a_shape;
    a_shape.push_back(batchCount);
    a_shape.push_back(argv[1]);
    a_shape.push_back(argv[2]);

    std::vector<size_t> b_shape;
    b_shape.push_back(batchCount);
    b_shape.push_back(argv[2]);
    b_shape.push_back(argv[3]);

    std::vector<size_t> c_shape;
    c_shape.push_back(batchCount);
    c_shape.push_back(argv[1]);
    c_shape.push_back(argv[3]);

    size_t size_A = cal_shape(a_shape);
    size_t mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    size_t size_B = cal_shape(b_shape);
    size_t mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    size_t mem_size_C = sizeof(float) * cal_shape(c_shape);
    
    // initialize host memory
    randInit(h_A, size_A);
    randInit(h_B, size_B);

    float *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));

    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
        int nIter = 10000;
        double total_time = 0;

        for (int j = 0; j < nIter; j++) 
        {
            float it_time;        cudaEventRecord(start);
            matmul(handle, d_A, a_shape, d_B, b_shape, d_C, c_shape, false, false);
                    cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&it_time, start, stop);        total_time += it_time;
        }


        double dSeconds = total_time/(double)nIter;
        printf("Time= %.3f(ms)\n", dSeconds);
    }
    checkError(cublasDestroy(handle), "cublasDestroy() error!\n");

}

int main(int argc, char* argv[])
{
    int shape[4] = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
    test(shape);
    return 0;
}

// nvcc matmul.cu -o matmul -lcublas -I /usr/local/cuda-9.0/samples/common/inc/ --std=c++11

