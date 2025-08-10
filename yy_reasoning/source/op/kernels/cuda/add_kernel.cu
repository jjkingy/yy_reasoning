#include "add_kernel.cuh"

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, 
    const float* __restrict__ in1,
    const float* __restrict__ in2, 
    float* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pack_num = size / 4;
    int pack_off = pack_num * 4;

    if(tid >= pack_num) {
        return;
    }

    float4 v1 = reinterpret_cast<const float4*>(in1)[tid];
    float4 v2 = reinterpret_cast<const float4*>(in2)[tid];
    
    float4 result = make_float4(v1.x + v2.x, v1.y + v2.y,
                                v1.z + v2.z, v1.w + v2.w);

    reinterpret_cast<float4*>(out)[tid] = result;

    for(int i = tid + pack_off; i < size; i += blockDim.x) {
        out[i] = in1[i] + in2[i];
    }
}


void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);
    int32_t size = static_cast<int32_t>(input1.size());
    CHECK_EQ(size, input2.size());
    CHECK_EQ(size, output.size());
    int32_t thread_num = 256;
    int32_t pack_num = size / 4;
    int32_t block_num = (pack_num + thread_num - 1) / thread_num;
    if(stream) {
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
        add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    } else {
        add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                    const_cast<float*>(output.ptr<float>()));
    }

}

}