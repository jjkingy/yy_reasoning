#include "op/rope.h"
#include <cmath>
#include "kernels/cpu/rope_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      _dim(dim),
      _kv_dim(kv_dim),
      _head_size(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status RoPELayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);

  tensor::Tensor sin_cache = this->get_input(3);
  tensor::Tensor cos_cache = this->get_input(4);

  if (_device_type == base::DeviceType::kDeviceCUDA) {
    CHECK(_cuda_config != nullptr);
  }
  kernel::get_rope_kernel(_device_type)(_dim, _kv_dim, _head_size, input_q, input_k, input_pos,
                                        sin_cache, cos_cache,
                                        _cuda_config ? _cuda_config->stream : nullptr);
  return base::error::Success();
}

base::Status RoPELayer::check() const {
  // pos tensor
  auto status = check_tensor_with_dim(get_input(2), base::DeviceType::kDeviceCPU,
                                      base::DataType::kDataTypeInt32, 1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(1), device_type_, data_type_, _kv_dim);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(0), device_type_, data_type_, _dim);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op