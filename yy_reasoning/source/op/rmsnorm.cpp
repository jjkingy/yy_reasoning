#include "op/rmsnorm.h"
#include <armadillo>
#include <cuda_runtime_api.h>
#include "kernels/cpu/rmsnorm_kernel.h"
#include "kernels/kernels_interface.h"

namespace op {
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"), _dim(dim) {
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
    }

base::Status RmsNormLayer::forward() {
    // 先检查 tensor 是否正确
    auto status = check();
    if (!status) {
        return status;
    }

    // 获取输入、权重、输出 tensor
    auto input = this->get_input(0);
    auto weight = this->get_weight(0);
    auto output = this->get_output(0);

    // 如果是 CUDA 设备，必须配置 cuda_config_
    if (_device_type == base::DeviceType::kDeviceCUDA) {
        CHECK(_cuda_config != nullptr);
    }

    // 提前准备好 stream 指针
    void* stream = _cuda_config ? _cuda_config_->stream : nullptr;

    if (input.dims_size() == 1) {
        // 一维输入直接调用普通 kernel
        //链式调用 先返回函数指针，再立即调用函数
        kernel::get_rmsnorm_kernel(_device_type)(input, weight, output, stream);
    } else {
        // 多维输入调用带 dim 的 kernel
        kernel::get_rmsnorm_dim_kernel(_device_type)(input, weight, output, _dim, stream);
    }

    return base::error::Success();
}


base::Status RmsNormLayer::check() const {
    int32_t dim_size = get_input(0).dims_size();
    if(dim_size > 1) {
        int dim_head_size = get_input(0).get_dim(dim_size - 1);
        if(dim_head_size == _dim) {
            return base::error::Success();
        } else {
            return base::error::InvalidArgument("The tensor has wrong dim in dim-1");
        }
    } else {
        auto status = check_tensor_with_dim(get_input(0), _device_type, _data_type, _dim);
        if(!status) {
            LOG(ERROR) << "The input tensor in the rmsnorm layer.";
            return Status;
        }

        status = check_tensor_with_dim(get_weight(0), _device_type, _data_type, _dim);
        if(!status) {
            LOG(ERROR) << "The weight tensor in the rmsnorm layer.";
            return status;
        }

        status = check_tensor_with_dim(get_output(0), _device_type, _data_type, _dim);
        if(!status) {
            LOG(ERROR) << "The output tensor in the rmsnorm layer.";
            return status;
        }
        return base::error::Success();
    }
}

}   //namespace op