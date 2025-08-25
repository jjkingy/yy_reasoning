#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <float.h>

#define THREAD_PER_BLOCK 256
#define warpSize 32

namespace kernel {
// __device__ void warpReduce(float* sdata, size_t* sidx, int tid) {
//     if(sdata[tid] < sdata[tid + 32]) {
//         sdata[tid] = sdata[tid + 32];
//         sidx[tid] = sidx[32 + tid];
//     }
//     if(sdata[tid] < sdata[tid + 16]) {
//         sdata[tid] = sdata[tid + 16];
//         sidx[tid] = sidx[16 + tid];
//     }
//     if(sdata[tid] < sdata[tid + 8]) {
//         sdata[tid] = sdata[tid + 8];
//         sidx[tid] = sidx[8 + tid];
//     }
//     if(sdata[tid] < sdata[tid + 4]) {
//         sdata[tid] = sdata[tid + 4];
//         sidx[tid] = sidx[4 + tid];
//     }
//     if(sdata[tid] < sdata[tid + 2]) {
//         sdata[tid] = sdata[tid + 2];
//         sidx[tid] = sidx[2 + tid];
//     }
//     if(sdata[tid] < sdata[tid + 1]) {
//         sdata[tid] = sdata[tid + 1];
//         sidx[tid] = sidx[1 + tid];
//     }
// }

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

// 声明外部函数（你已经写好的）
__global__ void argmax_kernel_fp32_block(const float* in, size_t size, size_t* output_idx, float* out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ size_t sidx[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sdata[tid] = idx < size ? in[idx] : -FLT_MAX;
    sidx[tid] = idx < size ? idx : 0;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s) {
            if(sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[s + tid];
            }
        }
        __syncthreads();
    }
    if(tid < 32) {
        warpReduce(sdata, sidx);
    }

    if(tid == 0) {
        output_idx[blockIdx.x] = sidx[0];
        out[blockIdx.x] = sdata[0];
    }

}

__global__ void argmax_kernel_fp32_all(const float* in, size_t size, size_t* block_idx, size_t* output_idx) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ size_t sidx[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    sdata[tid] = tid < size ? in[tid] : -FLT_MAX;
    sidx[tid] = tid < size ? block_idx[tid] : 0;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            if(sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[s + tid];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        *output_idx = sidx[0];
    }

}

size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) {
    int thread_num = THREAD_PER_BLOCK;
    int block_num = (size + thread_num - 1) / thread_num;

    auto alloc_cu = 
        base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();
    
    size_t* block_index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t) * block_num));
    float* out = static_cast<float*>(alloc_cu->allocate(sizeof(float) * block_num));
    size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));

    size_t output_index = 0;
    

    if(!stream) {
        argmax_kernel_fp32_block<<<block_num, thread_num>>>(input_ptr, size, block_index, out);
        argmax_kernel_fp32_all<<<1, thread_num>>>(out, block_num, block_index, index);
        cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        argmax_kernel_fp32_block<<<block_num, thread_num, 0, _stream>>>(input_ptr, size, block_index, out);
        argmax_kernel_fp32_all<<<1, thread_num, 0, _stream>>>(out, block_num, block_index, index);
        cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, _stream);
    }
    return output_index;
}

}


