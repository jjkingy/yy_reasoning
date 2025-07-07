#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {

/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size, int size, float eps) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if(bid >= dim_size) {
        return;
    }

    float* block_in = in + bid * size;
    float* block_out = out + bid * size;
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    const int pack_off = pack_num * pack_size;

    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(block_in);
    for(int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }
    
    for(int i = tid + pack_off; i < size; i += blockDim.x) {
        sum += block_in[i] * block_in[i];
    }

    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    sum = BlockReduce(temp).Sum(sum);
    if(threadIdx.x == 0) {
        shared_val = sum;
    }

    __syncthreads();
    sum = shared_val;

    const float scale = rsqrt(sum / static_cast<float>(size) + eps);    //size是int类型 所以要用static_cast做类型转换
    
    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(block_out);
    for(int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i) = make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }
    
    for(int i = tid + pack_off; i < size; i += blockDim.x) {
        block_out[i] = wei[i] * block_in[i] * scale;
    }

}


template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {

    const int tid = threadIdx.x;

    //准备向量化读取
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    const int pack_off = pack_size * pack_num;

    //并行计算平方和
    float sum = 0.0f;
    //这里用reinterpred_cast已经把in指针转化为float4指针了，计量步长已经变成了4个float为一组
    float4* in_pack = reinterpret_cast<float4*>(in);    //记录in的起始地址
    for(int i = tid; i < pack_num; i += blockDim.x) {   //blockDim是一个块里有多少个thread
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    //处理末尾
    for(int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    //规约 所有线程把各自局部 sum 规约成一个总和，写到共享内存。
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);

    //实现对sum的broadcast操作, 使每个线程都能拿到sum
    if(threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i) =
            make_float4(scale * in_float4.x * wei_float4.x,
                        scale * in_float4.y * wei_float4.y,
                        scale * in_float4.z * wei_float4.z,
                        scale * in_float4.w * wei_float4.w);
    }
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        out[i] = wei[i] * in[i] * scale;
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tnesor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif

    const int32_t size = static_cast<int32_t>(input.size());
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int thread_num = 128;
    if(stream) {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<128><<<1, thread_num, 0, _stream>(in_ptr, wei_ptr, out_ptr, size, ops);
    } else {
        row_rmsnorm_f32<128><<<1, thread_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

    const float eps = 1e-6f;
    const int32_t total_size = static_cast<int32_t>(input.size());
    const int32_t size = input.get_dim(input.dims_size() - 1);  //最后一维大小 特征维度
    const int32_t dim_size = total_size / size; //总共有多少行，即多少个最后一维度特征

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    constexpr int threads_num = 128;
    if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
        row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
}


}   //namespace kernel 


