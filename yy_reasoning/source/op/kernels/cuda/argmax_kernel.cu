#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <float.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>


#define THREAD_PER_BLOCK 256

namespace kernel {

/*我的终极内核--------------------------------*/
__device__ void warpReduce(float* sdata, size_t* sidx) {
    int tid = threadIdx.x;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, sdata[tid], offset);
        size_t other_idx = __shfl_down_sync(0xFFFFFFFF, sidx[tid], offset);
        if (other_val > sdata[tid] || (other_val == sdata[tid] && other_idx <  sidx[tid])) {
            sdata[tid] = other_val;
            sidx[tid] = other_idx;
        }
    }
}


__global__ void  argmax_kernel_fp32_myblock(const float* in, size_t* output_idx) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  __shared__ size_t sindex[THREAD_PER_BLOCK];
  unsigned int idx = blockIdx.x * (2*blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  sdata[tid] = fmaxf(in[idx], in[idx + blockDim.x]);
  sindex[tid] = in[idx] > in[idx + blockDim.x] ? idx : idx + blockDim.x;
  __syncthreads();

  for(int s = blockDim.x / 2; s > 32; s >>= 1) {
    if(tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      sindex[tid] = sdata[tid] > sdata[tid + s] ? sindex[tid] : sindex[tid + s];
    }
    __syncthreads();
  }
  if(tid < 32) {
    warpReduce(sdata, sindex);
  }
  if(tid == 0) {
    output_idx[blockIdx.x] = sindex[tid];
  }
}

template <typename T1, typename T2>
__global__ void argmax_kernel_fp32_myall(const float* in, size_t* block_idx, size_t* output_idx) {
  // extern __shared__ float sdata[];
  // extern __shared__ size_t sindex[];
  extern __shared__ unsigned char smem[];
  T1* sdata = reinterpret_cast<T1*>(smem);
  T2* sindex = reinterpret_cast<T2*>(smem + blockDim.x * sizeof(T1));

  int tid = threadIdx.x;
  sdata[tid] = fmaxf(in[block_idx[tid]], in[block_idx[tid + blockDim.x]]);
  sindex[tid] = 
    in[block_idx[tid]] > in[block_idx[tid + blockDim.x]] ? block_idx[tid] : block_idx[tid + blockDim.x];
  __syncthreads();

  for(int s = blockDim.x / 2; s > 32; s >>= 1) {
    if(tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      sindex[tid] = sdata[tid] > sdata[tid + s] ? sindex[tid] : sindex[tid + s];
    }
    __syncthreads();
  }
  if(tid < 32) {
    warpReduce(sdata, sindex);
  }
  if(tid == 0) {
    output_idx[tid] = sindex[tid];
  }
}


size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) {
    int thread_num = THREAD_PER_BLOCK;
    int block_num = (size + thread_num - 1) / thread_num;
    block_num /= 2;

    auto alloc_cu = 
        base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();
    
    size_t* d_block_idx = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t) * block_num));
    size_t* d_idx = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));

    size_t output_index = 0;
    int threads = block_num / 2;
    size_t smemSize = sizeof(float) * threads + sizeof(size_t) * threads;

    if(!stream) {
        argmax_kernel_fp32_myblock<<<block_num, THREAD_PER_BLOCK>>>(input_ptr, d_block_idx);
        argmax_kernel_fp32_myall<float, size_t><<<1, threads, smemSize>>>(input_ptr, d_block_idx, d_idx);
        cudaMemcpy(&output_index, d_idx, sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        argmax_kernel_fp32_myblock<<<block_num, THREAD_PER_BLOCK, 0, _stream>>>(input_ptr, d_block_idx);
        argmax_kernel_fp32_myall<float, size_t><<<1, threads, smemSize, _stream>>>(input_ptr, d_block_idx, d_idx);
        cudaMemcpyAsync(&output_index, d_idx, sizeof(size_t), cudaMemcpyDeviceToHost, _stream);
    }
    return output_index;
}

}    //kernel


