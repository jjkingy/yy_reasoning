#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "op/layer.h"
#include "kernels/kernels_interface.h"

namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size)
    :_dim(dim),
    _seq_len(seq_len),
    _vocab_size(vocab_size),
    LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
    reset_weight_size(1);
    reset_input_size(2);
    reset_output_size(1);
}


//仍有疑问，input12分别是什么
base::Status EmbeddingLayer::check() const {
    const auto& input_tensor = get_input(0);

    //存在疑问
    const auto& token_size = get_input(1).size();
    if(token_size > input_tensor.size()) {
        return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
    }

    base::Status status = check_tensor_with_dim(input_tensor, base::DeviceType::kDeviceCPU,
                                                base::DataType::kDataTypeInt32, token_size);
    if(!status) {
        LOG(ERROR) << "Input tensor shape mismatch. expected: [" << token_size
           << "], actual: " << input_tensor.shape().to_string();
        return status;
    }

    status = check_tensor_with_dim(get_weight(0), _device_type, _data_type, _vocab_size, _dim);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), _device_type, _data_type, token_size, _dim);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
    base::Status status = check();
    if(!status) {
        return status;
    }

    if(_device_type == base::DeviceType::kDeviceCUDA) {
        CHECK(_cuda_config != nullptr);
    }

    // 获取输入、权重、输出 tensor
    auto& input = this->get_input(0);
    auto& weight = this->get_weight(0);
    auto& output = this->get_output(0);
    void* stream = _cuda_config ? _cuda_config->stream : nullptr;

    kernel::get_emb_kernel(_device_type)(input, weight, output, _vocab_size, stream);

    return base::error::Success();
}

}