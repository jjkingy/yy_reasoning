#include "op/mha.h"
#include "kernels/cpu/mha_kernel.h"
#include "kernels/kernels_interface.h"

namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                       int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      _layer_index(layer_index),
      _kv_mul(kv_mul),
      _kv_dim(kv_dim),
      _seq_len(seq_len),
      _head_num(head_num),
      _head_size(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status MultiHeadAttention::forward() {
    auto state = check();
    if(!state) {
        return state;
    }
    const tensor::Tensor& mha_out = this->get_output(0);
    const tensor::Tensor& query_tensor = this->get_input(0);
    const tensor::Tensor& score_tensor = this->get_input(1);
    const tensor::Tensor& key_cache_tensor = this->get_input(2);
    const tensor::Tensor& value_cache_tensor = this->get_input(3);

    if (_device_type == base::DeviceType::kDeviceCUDA) {
        CHECK(_cuda_config != nullptr);
    }
    kernel::get_mha_kernel(_device_type)(_pos, _head_num, _layer_index, _seq_len, _kv_dim, _kv_mul,
                                        _head_size, mha_out, query_tensor, score_tensor, key_cache_tensor,
                                        value_cache_tensor, _device_type, _cuda_config ? _cuda_config.get() : nullptr);
    return state::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) {
    this->_pos = pos;
}

void MultiHeadAttention::set_layer_index(int32_t layer_idx) {
    this->_layer_index = layer_idx;
}

base::Status MultiHeadAttention::check() const {
    base::Status status;
    const int32_t input_tensor_num = 4;
    for(int i = 0; i < input_tensor_num; i++) {
        status = check_tensor(get_input(i), _device_type, _data_type);
        if (!status) {
        LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
        return status;
        }
    }
}


}   //op