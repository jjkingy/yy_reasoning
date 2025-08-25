#pragma once
#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {
void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size,
                       void* stream = nullptr);

}  // namespace kernel