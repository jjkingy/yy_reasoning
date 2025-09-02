#pragma once
#include <base/cuda_config.h>
#include "tensor/tensor.h"


namespace kernel {
/*      使用c++11新特性 using 代替 typedef*/

// typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream);
using AddKernel = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2, 
                            const tensor::Tensor& output, void* stream);

using MatmulKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, float scale, const CudaConfig* config);

using MatmulKernelQuant = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                    const tensor::Tensor& output, int32_t group_size,
                                    const tensor::Tensor& scale, const CudaConfig* config);

using RMSNormKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight, 
                                const tensor::Tensor& output, void* stream);

using RMSNormKernelDim = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t dim, void* stream);

using EmbeddingKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

using MHAKernel = void (*)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                          CudaConfig*);

using SwigluKernel = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              const tensor::Tensor& output, void* stream);

using RoPEKernel = void (*)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

MatmulKernel get_matmul_kernel(base::DeviceType device);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

AddKernel get_add_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);

}   //namespace kernel