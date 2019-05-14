#include "layernorm.cuh"

#include "shfl.cuh"
#include "../utils/common.h"
#include "../utils/manager.cuh"

template<typename U> __device__ U rsqrt(U v) {
  return U(1) / sqrt(v);
}

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}


template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1*n2;
    int l = 4*thrx;
    for (;  l+3 < n2;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      }
    }
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      U muB = WARP_SHFL(mu, srcLaneB);
      U countB = WARP_SHFL(count, srcLaneB);
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y] = mu;
          ubuf[2*wrt_y+1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2*threadIdx.y];
          U sigma2B = ubuf[2*threadIdx.y+1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
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
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2/U(n2), 0);
    }
  }
}

template<typename T, typename U> __global__
void cuApplyLayerNorm(
  T* __restrict__ output_vals,
  T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const T* __restrict__ gamma,
  const T* __restrict__ beta,
  const T* merge_add
  ) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  if(merge_add != nullptr){
    for(int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y){
      for(int i = threadIdx.y * blockDim.x  + threadIdx.x ; i < n2; i += blockDim.y * blockDim.x)
        vals[i + i1 * n2] += merge_add[i + i1 * n2];
    }
    __syncthreads();
  }

  for(int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y){
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf);
    const T* lvals = vals + i1*n2;
    T* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt<U>(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * (curr - mu));
      }
    }
  }
}

template<typename T> 
void op_LayerNorm::forward(
    T* output,
    T* input,
    size_t n1,
    size_t n2,
    T* merge_add
    )
{
    // auto stream TODO(): Muti-Stream 
    const dim3 threads(32,4,1);
    const dim3 blocks(1,min((long)65535,n1),1);
    int nshared = 
        threads.y > 1 ? 
	    threads.y*sizeof(T)+(threads.y/2)*sizeof(T) : 
	    0;
    cuApplyLayerNorm<<<blocks, threads, nshared, handle->cal_stream>>>(
		    output,
		    input,
		    n1,n2,
		    T(epsilon),
          gamma,beta, merge_add);
}

template
void op_LayerNorm::forward<float>(
    float* output,
    float* input,
    size_t n1,
    size_t n2,
    float* merge_add
    );
