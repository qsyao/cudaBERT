#include "matmul.cuh"

void deviceMatmul(cublasHandle_t handle, float *d_A, 
                std::vector<size_t> a_shape, 
                float *d_B,
                std::vector<size_t> b_shape, 
                float *d_C, 
                bool transpose_a_, 
                bool transpose_b_, 
                const float alpha, 
                const float beta,
                const int batchCount, 
                const long long int strideA, 
                const long long int strideB, 
                const long long int strideC) {
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

    if (batchCount == -1)
    {
        cublasStatus_t ret =
            cublasSgemm(handle, 
                        blas_transpose_b, 
                        blas_transpose_a,
                        n, 
                        m, 
                        k, 
                        &alpha, 
                        d_B, 
                        transpose_b_ ? k : n, 
                        d_A,
                        transpose_a_ ? m : k, 
                        &beta, 
                        d_C, 
                        n);
        checkError(ret, "cublas Sgemm returned an error!\n");
    }
    else
    {
        cublasStatus_t ret =
            cublasSgemmStridedBatched(handle, 
                                    blas_transpose_b, 
                                    blas_transpose_a,
                                    n, 
                                    m, 
                                    k, 
                                    &alpha, 
                                    d_B, 
                                    transpose_b_ ? k : n, 
                                    strideB, 
                                    d_A, 
                                    transpose_a_ ? m : k, 
                                    strideA, 
                                    &beta, 
                                    d_C, 
                                    n, 
                                    strideC, 
                                    batchCount);
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

void matmul(cublasHandle_t handle, 
            float *d_A, 
            std::vector<size_t> a_shape, 
            float *d_B,
            std::vector<size_t> b_shape, 
            float *d_C, 
            std::vector<size_t> c_shape, 
            bool transpose_a_, 
            bool transpose_b_, 
            const float alpha, 
            const float beta, 
            long long int custom_strideA)
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
        if (custom_strideA != -1)
            strideA = (long long int)custom_strideA;
        long long int strideB = (long long int)(cal_shape(b_shape) / batchCount);
        long long int strideC = (long long int)(cal_shape(c_shape) / batchCount);
        deviceMatmul(handle, 
                    d_A, 
                    a_shape, 
                    d_B, 
                    b_shape, 
                    d_C, 
                    transpose_a_, 
                    transpose_b_, 
                    alpha, 
                    beta, 
                    batchCount, 
                    strideA, strideB, strideC);
    }
    return ;
}

void randInit(float *data, int size)
{
    for (int i = 0; i < size; ++i) 
    {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        // data[i] = 1.1;
    }
}

/*
int main()
{
    test();
}
*/
// nvcc matmul.cu -o matmul -lcublas -I /usr/local/cuda-9.0/samples/common/inc/ --std=c++11

