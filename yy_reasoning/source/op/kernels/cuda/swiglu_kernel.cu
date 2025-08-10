#include "tensor/tensor.h"
#include "swiglu_kernel.cuh"

namespace kernel {

__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * 4;
    
    int pack_num = size / 4;
    int pack_off = pack_num * 4;

    if(tid >= pack_num) {
        return;
    }

    float4 v1 = reinterpret_cast<const float4*>(in1)[tid];
    float4 v2 = reinterpret_cast<const float4*>(in2)[tid];
    float4 result;

    result.x = v1.x * (1.0f / (1.0f + __expf(-v1.x))) * v2.x;
    result.y = v1.y * (1.0f / (1.0f + __expf(-v1.y))) * v2.y;
    result.z = v1.z * (1.0f / (1.0f + __expf(-v1.z))) * v2.z;
    result.w = v1.w * (1.0f / (1.0f + __expf(-v1.w))) * v2.w;
    reinterpret_cast<float4*>(out)[tid] = result;

    for(int i = pack_off + tid; i < size; i += blockDim.x) {
        float v1 = in1[i];
        float v2 = in2[i];
        float sig = 1.0f / (1.0f + __expf(-v1));
        out[i] = v1 * sig * v2;
    }
}


void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK_EQ(input2.is_empty(), false);
    CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK_EQ(output.is_empty(), false);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);   

    int size = static_cast<int32_t>(input1.size());
    int num_threads = 128;
    int pack_num = size / 4;
    int num_blocks = (pack_num + num_threads - 1) / num_threads;

    const float* in1 = input1.ptr<float>();
    const float* in2 = input2.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>()); 


    if(!stream) {
        swiglu_kernel_cu_fp32<<<num_blocks, num_threads>>>(
            size, in1, in2, out);
    } else {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        swiglu_kernel_cu_fp32<<<num_blocks, num_threads, 0, _stream>>>(
            size, in1, in2, out);
    }
    
}

}   //kernel