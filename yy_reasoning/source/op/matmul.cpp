#include "op/matmul.h"
#include "kernels/kernels_interface.h"
#include "kernels/cpu/matmul_kernel.h"

namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
    dim0(dim0), dim1(dim1), _has_bias(has_bias) {
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
        if(_has_bias) {
            _bias.resize(1);
        }
}

base::Status MatmulLayer::check() const {
    auto status = check_tensor_with_dim(get_input(0), _device_type, _data_type, _dim1);
    if(!status) {
        LOG(ERROR) << "The input tensor error in the matmul layer";
        return status;
    }

    if(!_is_quant_layer) {
        status = check_tensor_with_dim(get_weight(0), _device_type, _data_type, _dim0, _dim1);
        if(!status) {
            LOG(ERROR) << "The weight tensor error in the matmul layer";
            return status;
        }
    } else {
        status = check_tensor_with_dim(get_weight(0), _device_type, base::DataType::kDataTypeInt8, 
                                        _dim0, _dim1);
        if(!status) {
            LOG(ERROR) << "The weight tensor error in the matmul layer";
            return status;
        }
    }

    //如果是量化模型还需要检查scales因子
    if(_is_quant_layer) {
        status = check_tensor_with_dim(_scales, _device_type, base::DataType::kDataTypeFp32, _scales.size());
        if(!status) {
            LOG(ERROR) << "The scale tensor error in the matmul layer";
            return status;
        }
    }

    status = check_tensor_with_dim(get_output(0), _device_type, _data_type, _dim0);
    if(!status) {
        LOG(ERROR) << "The output tensor error in the matmul layer";
        return status;
    }

    return base::error::Success();
}

// //forward未完成 先把后端写完再回来补充
// base::Status MatmulLayer::forward() {
//     auto status = check();
//     if(!status) {
//         return Status;
//     }
//     if(_device_type = base::DeviceType::kDeviceCUDA) {
//         CHECK(_cuda_config != nullptr);
//     }
//     if(_is_quant_layer) {
//         kernel::get_matmul_kernel_quant8(_device_type)(get_input(0), get_weight(0), get_output(0),
//                                                         _group_size, _scalse, _cuda_config ? _cuda_config.get() : nullptr);
//     } else {
//         kernel::
//     }
// }

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr, 
                                    base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _bias.size());
    CHECK_NE(bias_ptr, nullptr);
    
    //创建Buffer
    size_t size = dim * sizeof(float);
    std::shared_ptr<base::Buffer> buffer = 
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);

    if(device_type != base::DeviceType::kDeviceUnknown) {
        buffer->set_device_type(device_type);
    }

    if(!_is_quant_layer) {
        tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));  //buffer数据赋值给tensor
        _bias.at(idx) = bias;   //存入成员变量
    } else {
        //quant layer
        tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));
        _bias.at(idx) = bias;

        const int32_t bias_size = static_cast<int32_t>(bias.size());
        CHECK(bias_size % _group_size == 0);

        //|------ bias int8 (bias_size 个元素) ------|------ scales float (scale_nums 个元素) ------|
        int32_t scale_nums = bias_size / _group_size;
        //bias_size 是 bias 元素个数，在 int8 模型里一个元素 = 1 字节
        //因此 (int8_t*)bias_ptr + bias_size 正好指向 scale 起始地址。
        _scales = tensor::Tensor(base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                                reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size))    //scale数据跟在bias后面
        _scales.set_device_type(device_type);
    }
    
    return base::error::Success();
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const  {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _bias.size());
    return _bias.at(idx);
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _bias.size());
    return _bias.at(idx);
}

void MatmulLayer::to_cuda() {
    LayerParam::to_cuda();
    if(_has_bias) {
        for(auto& bias: _bias) {
            bias.to_cuda(_cuda_config ? _cuda_config->stream : nullptr);
        }
    }
}

}   //namespace op