#include "op/add.h"
#include "kernels/kernels_interface.h"

namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
    reset_input_size(2);
    reset_output_size(1);
}

base::Status VecAddLayer::check() const {
    tensor::Tensor input1 = this->get_input(0);
    tensor::Tensor input2 = this->get_input(1);
    size_t size = input1.size();

    base::Status status = check_tensor_with_dim(input1, _device_type, _data_type, size);
    if(!status) {
        LOG(ERROR) << "The input tensor1 error in add layer";
        return status;
    }

    status = check_tensor_with_dim(input2, _device_type, _data_type, size);
    if(!status) {
        LOG(ERROR) << "The input tensor2 error in add layer";
        return status;
    }

    tensor::Tensor output = this->get_output(0);
    status = check_tensor_with_dim(output, _device_type, _data_type, size);
    if(!status) {
        LOG(ERROR) << "The input tensor2 error in add layer";
        return status;
    }
    return base::error::Success();
}

base::Status VecAddLayer::forward() {
    auto status = this->check();
    if(!status) {
        return status;
    }
    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);
    if(_device_type == base::DeviceType::kDeviceCUDA) {
        CHECK(_cuda_config != nullptr);
    }

    kernel::get_add_kernel(_device_type)(input1, input2, output, _cuda_config ? _cuda_config->stream : nullptr);

    return base::error::Success();
}

}   //namespace op

