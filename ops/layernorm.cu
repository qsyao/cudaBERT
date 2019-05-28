#include "layernorm.cuh"

#include "shfl.cuh"
#include "../utils/common.h"
#include "../utils/manager.cuh"

template<typename U> __device__ U rsqrt(U v) {
    return U(1) / sqrt(v);
}

template<typename U>
__device__
void cuWelfordOnlineSum(
        const U curr,
        U &mu,
        U &sigma2,
        U &count) {
    count = count + U(1);
    U delta = curr - mu;
    U lmean = mu + delta / count;
    mu = lmean;
    U delta2 = curr - lmean;
    sigma2 = sigma2 + delta * delta2;
}


template<typename U>
__device__
void cuChanOnlineSum(
        const U muB,
        const U sigma2B,
        const U countB,
        U &mu,
        U &sigma2,
        U &count) {
    U delta = muB - mu;
    U nA = count;
    U nB = countB;
    count = count + countB;
    U nX = count;
    if (nX > U(0)) {
        nA = nA / nX;
        nB = nB / nX;
        mu = nA * mu + nB * muB;
        sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
    } else {
        mu = U(0);
        sigma2 = U(0);
    }
}

template<typename T, typename U>
__device__
void cuWelfordMuSigma2(
        const T *__restrict__ vals,
        const int n1,
        const int n2,
        const int i1,
        U &mu,
        U &sigma2,
        U *buf) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
    //
    // compute variance and mean over n2
    U count = U(0);
    mu = U(0);
    sigma2 = U(0);
    if (i1 < n1) {
        // one warp normalizes one n1 index,
        // synchronization is implicit
        // initialize with standard Welford algorithm
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        const T *lvals = vals + i1 * n2;
        int l = 4 * thrx;
        for (; l + 3 < n2; l += 4 * numx) {
            for (int k = 0; k < 4; ++k) {
                U curr = static_cast<U>(lvals[l + k]);
                cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
            }
        }
        for (; l < n2; ++l) {
            U curr = static_cast<U>(lvals[l]);
            cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
        }
        // intra-warp reductions
        for (int l = 0; l <= 4; ++l) {
            int srcLaneB = (threadIdx.x + (1 << l)) & 31;
            U muB = WARP_SHFL(mu, srcLaneB);
            U countB = WARP_SHFL(count, srcLaneB);
            U sigma2B = WARP_SHFL(sigma2, srcLaneB);
            cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
        }
        // threadIdx.x == 0 has correct values for each warp
        // inter-warp reductions
        if (blockDim.y > 1) {
            U *ubuf = (U *) buf;
            U *ibuf = (U *) (ubuf + blockDim.y);
            for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                    const int wrt_y = threadIdx.y - offset;
                    ubuf[2 * wrt_y] = mu;
                    ubuf[2 * wrt_y + 1] = sigma2;
                    ibuf[wrt_y] = count;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.x == 0 && threadIdx.y < offset) {
                    U muB = ubuf[2 * threadIdx.y];
                    U sigma2B = ubuf[2 * threadIdx.y + 1];
                    U countB = ibuf[threadIdx.y];
                    cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
                }
                __syncthreads();
            }
            // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                ubuf[0] = mu;
                ubuf[1] = sigma2;
            }
            __syncthreads();
            mu = ubuf[0];
            sigma2 = ubuf[1] / U(n2);
            // don't care about final value of count, we know count == n2
        } else {
            mu = WARP_SHFL(mu, 0);
            sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
        }
    }
}

template<typename T, typename U>
__global__
void cuApplyLayerNorm(
        T *__restrict__ output_vals,
        U *__restrict__ mean,
        U *__restrict__ invvar,
        T *__restrict__ vals,
        const int n1,
        const int n2,
        const U epsilon,
        const T *__restrict__ gamma,
        const T *__restrict__ beta,
        const T *merge_add,
        T *__restrict__ stored_input
) {
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensors are contiguous
    //
    if (merge_add != nullptr) {
        for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
            for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < n2; i += blockDim.y * blockDim.x) {
                vals[i + i1 * n2] += merge_add[i + i1 * n2];
                if(stored_input != nullptr)
                    stored_input[i + i1 * n2] = vals[i + i1 * n2];
            }
        }
        __syncthreads();
    }
    else if(stored_input != nullptr){
        for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y)
            for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < n2; i += blockDim.y * blockDim.x)
                if(stored_input != nullptr)
                    stored_input[i + i1 * n2] = vals[i + i1 * n2];
        __syncthreads();
    }

    for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
        SharedMemory<U> shared;
        U *buf = shared.getPointer();
        U mu, sigma2;
        cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
        const T *lvals = vals + i1 * n2;
        T *ovals = output_vals + i1 * n2;
        U c_invvar = rsqrt<U>(sigma2 + epsilon);
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        if (gamma != NULL && beta != NULL) {
            for (int i = thrx; i < n2; i += numx) {
                U curr = static_cast<U>(lvals[i]);
                ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
            }
        } else {
            for (int i = thrx; i < n2; i += numx) {
                U curr = static_cast<U>(lvals[i]);
                ovals[i] = static_cast<T>(c_invvar * (curr - mu));
            }
        }
        if(mean != nullptr) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                mean[i1] = mu;
                invvar[i1] = c_invvar;
            }
        }
    }
}

template<typename T>
void op_LayerNorm::forward(
        T *&output,
        T *input,
        size_t n1,
        size_t n2,
        T *merge_add
) {
//    debug_tensor_gpu<float>(std::string("gamma"), input, 1, n1, 1);

    if(handle->is_train) {
        mean = handle->global_malloc_manage_float.get_new_head_point(n1);
        invvar = handle->global_malloc_manage_float.get_new_head_point(n1);
        stored_input = handle->global_malloc_manage_float.get_new_head_point(n1 * n2);
        output = handle->global_malloc_manage_float.get_new_head_point(n1 * n2);
    }
    else {
        mean = nullptr;
        invvar = nullptr;
        stored_input = nullptr;
    }

    // auto stream TODO(): Muti-Stream 
    const dim3 threads(32, 4, 1);
    const dim3 blocks(1, min((long) 65535, n1), 1);
    int nshared =
            threads.y > 1 ?
            threads.y * sizeof(T) + (threads.y / 2) * sizeof(T) :
            0;
    cuApplyLayerNorm << < blocks, threads, nshared, handle->cal_stream >> > (
            output, mean, invvar,
                    input,
                    n1, n2,
                    T(epsilon),
                    gamma, beta, merge_add, stored_input);

}

template
void op_LayerNorm::forward<float>(
        float *&output,
        float *input,
        size_t n1,
        size_t n2,
        float *merge_add
);


template<typename T>
__device__ void cuLoadWriteStridedInputs(
        const int i1_block,
        const int thr_load_row_off,
        const int thr_load_col_off,
        const int i2_off,
        const int row_stride,
        T *warp_buf1,
        T *warp_buf2,
        const T *input,
        const T *dout,
        const int i1_end,
        const int n2,
        const T *__restrict__ mean,
        const T *__restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        T curr_mean = mean[i1];
        T curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                T curr_input = static_cast<T>(input[load_idx]);
                T curr_dout = static_cast<T>(dout[load_idx]);
                warp_buf1[write_idx] = curr_dout;
                warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
            } else {
                warp_buf1[write_idx] = T(0);
                warp_buf2[write_idx] = T(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf1[write_idx] = T(0);
            warp_buf2[write_idx] = T(0);
        }
    }
}

template<typename T>
__device__ void cuLoadAddStridedInputs(
        const int i1_block,
        const int thr_load_row_off,
        const int thr_load_col_off,
        const int i2_off,
        const int row_stride,
        T *warp_buf1,
        T *warp_buf2,
        const T *input,
        const T *dout,
        const int i1_end,
        const int n2,
        const T *__restrict__ mean,
        const T *__restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        T curr_mean = mean[i1];
        T curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                T curr_input = static_cast<T>(input[load_idx]);
                T curr_dout = static_cast<T>(dout[load_idx]);
                warp_buf1[write_idx] += curr_dout;
                warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
            }
        }
    }
}

template<typename T>
__global__ void cuComputePartGradGammaBeta(
        const T *__restrict__ dout,
        const T *__restrict__ input,
        const int n1,
        const int n2,
        const T *__restrict__ mean,
        const T *__restrict__ invvar,
        T epsilon,
        T *part_grad_gamma,
        T *part_grad_beta) {

    const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;

    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<T> shared;
    T *buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    T *warp_buf1 = (T *) buf;
    T *warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2,
                             input, dout, i1_end, n2, mean, invvar);
    for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end; i1_block += blockDim.y * blockDim.y) {
        cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2,
                               input, dout, i1_end, n2, mean, invvar);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    T acc1 = T(0);
    T acc2 = T(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k * blockDim.y;
        int idx1 = row1 * row_stride + threadIdx.x;
        acc1 += warp_buf1[idx1];
        acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1 * row_stride + threadIdx.x;
            int idx2 = row2 * row_stride + threadIdx.x;
            warp_buf1[idx1] += warp_buf1[idx2];
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename T>
__global__ void cuComputeGradGammaBeta(
        const T *part_grad_gamma,
        const T *part_grad_beta,
        const int part_size,
        const int n1,
        const int n2,
        T *grad_gamma,
        T *grad_beta) {
    // sum partial gradients for gamma and beta
    SharedMemory<T> shared;
    T *buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        int num_warp_reductions = part_size / blockDim.y;
        T sum_gamma = T(0);
        T sum_beta = T(0);
        const T *part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
        const T *part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
        for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
            sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
            sum_beta += part_grad_beta_ptr[warp_offset * n2];
        }
        // inter-warp reductions
        const int nbsize3 = blockDim.x * blockDim.y / 2;
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
                buf[write_idx + nbsize3] = sum_beta;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
                sum_beta += buf[read_idx + nbsize3];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
            grad_beta[i2] = sum_beta;
        }
    }
}

template<typename T>
__global__ void cuComputeGradInput(
        const T *__restrict__ dout,
        const T *__restrict__ input,
        const int n1,
        const int n2,
        const T *__restrict__ mean,
        const T *__restrict__ invvar,
        T epsilon,
        const T *gamma,
        T *grad_input) {
    int i1 = blockIdx.y;
    if (i1 < n1) {
        T sum_loss1 = T(0);
        T sum_loss2 = T(0);
        const T c_mean = mean[i1];
        const T c_invvar = invvar[i1];
        const T *k_input = input + i1 * n2;
        const T *k_dout = dout + i1 * n2;
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        if (gamma != NULL) {
            int l = 4 * thrx;
            for (; l + 3 < n2; l += 4 * numx) {
                for (int k = 0; k < 4; ++k) {
                    const T c_h = static_cast<T>(k_input[l + k]);
                    const T c_loss = static_cast<T>(k_dout[l + k]);
                    sum_loss1 += c_loss * gamma[l + k];
                    sum_loss2 += c_loss * gamma[l + k] * (c_h - c_mean) * c_invvar;
                }
            }
            for (; l < n2; ++l) {
                const T c_h = static_cast<T>(k_input[l]);
                const T c_loss = static_cast<T>(k_dout[l]);
                sum_loss1 += c_loss * gamma[l];
                sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
            }
        } else {
            int l = 4 * thrx;
            for (; l + 3 < n2; l += 4 * numx) {
                for (int k = 0; k < 4; ++k) {
                    const T c_h = static_cast<T>(k_input[l + k]);
                    const T c_loss = static_cast<T>(k_dout[l + k]);
                    sum_loss1 += c_loss;
                    sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
                }
            }
            for (; l < n2; ++l) {
                const T c_h = static_cast<T>(k_input[l]);
                const T c_loss = static_cast<T>(k_dout[l]);
                sum_loss1 += c_loss;
                sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            }
        }
        // intra-warp reductions
        for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
            sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
            sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
        }
        // inter-warp reductions
        if (blockDim.y > 1) {
            SharedMemory<T> shared;
            T *buf = shared.getPointer();
            for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                    const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                    buf[2 * wrt_i] = sum_loss1;
                    buf[2 * wrt_i + 1] = sum_loss2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.y < offset) {
                    const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
                    sum_loss1 += buf[2 * read_i];
                    sum_loss2 += buf[2 * read_i + 1];
                }
                __syncthreads();
            }
            if (threadIdx.y == 0) {
                buf[2 * threadIdx.x] = sum_loss1;
                buf[2 * threadIdx.x + 1] = sum_loss2;
            }
            __syncthreads();
            if (threadIdx.y != 0) {
                sum_loss1 = buf[2 * threadIdx.x];
                sum_loss2 = buf[2 * threadIdx.x + 1];
            }
        }
        // all threads now have the two sums over l
        T fH = (T) n2;
        T term1 = (T(1) / fH) * c_invvar;
        T *k_grad_input = grad_input + i1 * n2;
        if (gamma != NULL) {
            for (int l = thrx; l < n2; l += numx) {
                const T c_h = static_cast<T>(k_input[l]);
                const T c_loss = static_cast<T>(k_dout[l]);
                T f_grad_input = fH * c_loss * gamma[l];
                f_grad_input -= sum_loss1;
                f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<T>(f_grad_input);
            }
        } else {
            for (int l = thrx; l < n2; l += numx) {
                const T c_h = static_cast<T>(k_input[l]);
                const T c_loss = static_cast<T>(k_dout[l]);
                T f_grad_input = fH * c_loss;
                f_grad_input -= sum_loss1;
                f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<T>(f_grad_input);
            }
        }
    }
}

void op_LayerNorm::update_weights(size_t n) {
    if(handle->update_learning_rate) {
        learning_rate = handle->learning_rate;
    }
    if (handle->optim_method == "sgd") {
        if (gamma != NULL && beta != NULL) {
            apply_sgd_running_time(gamma, grad_gamma, n, learning_rate, handle);
            apply_sgd_running_time(beta, grad_beta, n, learning_rate, handle);
        }
    }
    else if(handle->optim_method == "adam") {
        if (gamma != NULL && beta != NULL) {
            apply_adam_running_time(gamma, grad_gamma, n, gamma_m_t, gamma_v_t, beta_1_t,
                                    beta_2_t, handle, learning_rate, weight_decay_rate, beta_1,
                                    beta_2, adam_epsilon, step);

            apply_adam_running_time(beta, grad_beta, n, beta_m_t, beta_v_t, beta_1_t,
                                    beta_2_t, handle, learning_rate, weight_decay_rate, beta_1,
                                    beta_2, adam_epsilon, step);

            beta_1_t = beta_1_t * beta_1;
            beta_2_t = beta_2_t * beta_2;
            step += 1;
        }
    }
    else if(handle->optim_method == "momentum") {
        if (gamma != NULL && beta != NULL) {
            apply_momentum_running_time(gamma, grad_gamma, n, momentum_gamma_v, learning_rate, momentum_beta, handle, step);
            apply_momentum_running_time(beta, grad_beta, n, momentum_beta_v, learning_rate, momentum_beta, handle, step);
            step += 1;
        }
    }
}

template<typename T>
void op_LayerNorm::backward(T *dout, size_t n1, size_t n2) {
    grad_input = handle->global_malloc_manage_float.get_new_head_point(n1 * n2);
    if(gamma != NULL && beta != NULL) {
        grad_gamma = handle->global_malloc_manage_float.get_new_head_point(n2);
        grad_beta = handle->global_malloc_manage_float.get_new_head_point(n2);
    }
    if (gamma != NULL && beta != NULL) {
        // compute grad_gamma(j) and grad_beta(j)

        const int part_size = 16;
        const dim3 threads2(32, 4, 1);
        const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
        const int nshared2_a = 2 * sizeof(T) * threads2.y * threads2.y * (threads2.x + 1);
        const int nshared2_b = threads2.x * threads2.y * sizeof(T);
        const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
        float *part_grad_gamma = handle->global_malloc_manage_float.get_new_head_point(part_size * n2);
        float *part_grad_beta = handle->global_malloc_manage_float.get_new_head_point(part_size * n2);
        cuComputePartGradGammaBeta << < blocks2, threads2, nshared2, handle->cal_stream >> > (
                        dout,
                        stored_input,
                        n1, n2,
                        mean,
                        invvar,
                        T(epsilon),
                        part_grad_gamma,
                        part_grad_beta);

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(T);
        cuComputeGradGammaBeta << < blocks3, threads3, nshared3, handle->cal_stream >> > (
                part_grad_gamma,
                        part_grad_beta,
                        part_size,
                        n1, n2,
                        grad_gamma,
                        grad_beta);
    }

    // compute grad_input
    const dim3 threads1(32, 4, 1);
    const dim3 blocks1(1, n1, 1);
    int nshared =
            threads1.y > 1 ? threads1.y * threads1.x * sizeof(T) : 0;
    cuComputeGradInput << < blocks1, threads1, nshared, handle->cal_stream >> > (
                    dout,
                    stored_input,
                    n1, n2,
                    mean,
                    invvar,
                    T(epsilon),
                    gamma,
                    grad_input);

    if (handle->optim_running_time)
        update_weights(n2);
}

template
void op_LayerNorm::backward(float *dout, size_t n1, size_t n2);