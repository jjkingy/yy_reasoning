#pragma once
#include "tensor/tensor.h"
#include "base/cuda_config.h"


namespace kernel {
/*      使用c++11新特性 using 代替 typedef*/

// typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream);
using AddKernel = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2, 
                            const tensor::Tensor& output, void* stream);

using MatmulKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, float scale, const CudaConfig* config);

using MatmulKernelQuant = void (*)(const tensor::Tensor& innput, const tensor::Tensor& weight,
                                    const tensor::Tensor& output, int32_t group_size,
                                    const tensor::Tensor& scale, const CudaConfig* config);

using RMSNormKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight, 
                                const tensor::Tensor& output, void* stream);

using RMSNormKernelDim = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t dim, void* stream);

using EmbeddingKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);


void softmax_inplace_cpu(const float* input_ptr, size_t size);

MatmulKernel get_matmul_kernel(base::DeviceType device);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type);

}   //namespace kernel