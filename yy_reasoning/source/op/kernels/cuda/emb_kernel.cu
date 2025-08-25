#include "emb_kernel.cuh"

namespace kernel {

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                const int32_t* in_ptr, const float* wei_ptr, float* out_ptr) {
    //token_idx表示当前输入的第idx个输入编号
    int32_t token_idx = blockIdx.x;
    if(token_idx >= token_num) {
        return;
    }
    //token表示当前idx个输入的具体token 取weight中token对应的weight_dim
    int32_t token = in_ptr[token_idx];
    if(token >= vocab_size) {
        return;
    }

    float* out_start = out_ptr + token_idx * weight_dim;
    const float* wei_start = wei_ptr + token * weight_dim;

    for(int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
        out_start[i] = wei_start[i];
    }
                 
}


void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
    tensor::Tensor input_cu;

    //如果不在gpu上则拷贝一份到gpu
    if(input.device_type() != base::DeviceType::kDeviceCUDA) {
        input_cu = input.clone();
        input_cu.to_cuda();
    }
    const int32_t input_num = static_cast<int32_t>(input.size());
    constint32_t weight_dim = weight.get_dim(1);

    CHECK(weight.device_type() == output.device_type());
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    constexpr int32_t max_seq_len = 512;
    constexpr int32_t thread_num = 128;
    int32_t* in_ptr = input_cu.ptr<int32_t>();
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    if (stream) {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, _stream>>>(vocab_size, input_num, weight_dim,
                                                                    in_ptr, wei_ptr, out_ptr);
    } else {
        emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                        wei_ptr, out_ptr);
    }
    
}

}   //kernel