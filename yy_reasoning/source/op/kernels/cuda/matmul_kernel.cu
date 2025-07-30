#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"

namespace kernel {

template<int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;

    //超过行号就返回
    if(start_row >= K) {
        return;
    }

    constexpr int pack_size = 4;
    const int pack_num = M / pack_size;
    const int pack_off = pack_num * pack_size;
    
    //遍历当前block负责的每一行
#pragma unroll
    for(int p = start_row; p < min(end_row, K); p++) {
        sdata[tid] = 0;

        int row_offset = p * M;
        //转换指针
        // float4* input_float_ptr = (float4*)input;
        // float4* weight_float_ptr = (float4*)(weight + row_offset);
        const float4* input_float_ptr  = reinterpret_cast<const float4*>(input);
        const float4* weight_float_ptr = reinterpret_cast<const float4*>(weight + row_offset);


#pragma unroll
        for(int i = tid; i < pack_num; i += blockDim.x) {
            float4 input_float4 = *(input_float_ptr + i);
            float4 weight_float4 = *(weight_float_ptr + i);

            float part_sum = input_float4.x * weight_float4.x
                            + input_float4.y * weight_float4.y
                            + input_float4.z * weight_float4.z
                            + input_float4.w * weight_float4.w;
            sdata[tid] += part_sum;
        }

        //处理不能被4整除的尾部 i起始位置pack_off + tid
        for(int i = pack_off + tid; i < M; i += blockDim.x) {
            sdata[tid] += input[i] * weight[row_offset + i];
        }

        //线程同步
        __syncthreads();

        //使用CUB工具 BLOCKREDUCE进行块内规约
        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;  //临时存储
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);

        // __syncthreads();

        if(tid == 0) {
            output[p] = part_sum;
        }
        
        // 再同步一次，保证写结果时不会影响下一次循环
        __syncthreads();

    }   

}

template<int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight, const float* scales,
                                          const int32_t group_size, float* output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;

    if(start_row >= K) {
        return;
    }

    //处理每一行
    for(int p = start_row; p < min(end_row, K); p++) {
        sdata[tid] = 0;
        
        for(int i = tid; i < M; i += THREAD_PER_BLOCK) {
            const int weight_idx = p * M  + i;
            const int group_idx = weight_idx / group_size;
            sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
        }
        __syncthreads();

        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);


        // __syncthreads();
        if(tid == 0) {
            output[p] = part_sum;
        }
        __syncthreads();
    }    

}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

    const int32_t K = weight.get_dim(0);    //row
    const int32_t M = weight.get_dim(1);    //col
    int pack_size = 4;
    CHECK_EQ(M % pack_size, 0);

    if(config && config->stream) {
        matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),M, K);
    } else {
        matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(
            input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),M, K);
    }

}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config){
    CHECK(config != nullptr);
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

    const int32_t K = weight.get_dim(0);    //row
    const int32_t M = weight.get_dim(1);    //col

    int pack_size = 4;
    CHECK_EQ(M % pack_size, 0);
    CHECK_EQ(M, input.get_dim(0));
    if(config->stream) {
        matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
            const_cast<float*>(output.ptr<float>()), M, K);
    } else {
        matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(), 
            scale.ptr<float>(), group_size, const_cast<float*>(output.ptr<float>()), M, K);
    }
}
    
}   