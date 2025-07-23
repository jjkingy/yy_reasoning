#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>


namespace op {
//BaseLayer
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name)
    :_device_type(device_type),
    _layer_type(layer_type),
    _data_type(data_type),
    _layer_name(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return _data_type; }

LayerType BaseLayer::layer_type() const { return _layer_type; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const {return _layer_name;}

void BaseLayer::set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }

base::DeviceType BaseLayer::device_type() const { return _device_type;}

void BaseLayer::set_device_type(base::DeviceType device_type) { _device_type = device_type; }

///Layer
Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const {
    if(tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor Parameter is empty.");
    }
    if(tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has wrong device type");
    }
    if(tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The data type is wrong");
    }
    return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type, 
                                        base::DataType data_type, ...) const {
    va_list args;

    if(tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor Parameter is empty.");
    }
    if(tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has wrong device type");
    }
    if(tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The data type is wrong");
    }

    va_start(args, data_type);
    int32_t ndim = tensor.dims_size();
    for(auto i = 0; i < ndim; i++) {
        int32_t dim = va_arg(args, int32_t);
        if(dim != tensor.get_dim(i)) {
            va_end(args);
            return base::error::InvalidArgument("The tensor has wrong dim in dim" + std::to_string(i));
        }
    }
    
    va_end(args);
    return base::error::Success();
}

base::Status Layer::check() {
    return base::error::FunctionNotImplement("The check function is not implent yet");
}

base::Status Layer::forward() {
    return base::error::FunctionNotImplement("");
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
    CHECK_GE(idx, 0);           // 检查 idx >= 0，防止负下标
    CHECK_LT(idx, _inputs.size()); // 检查 idx < _inputs.size()，防止越界
    this->_inputs.at(idx) = input;
}
  
// set_output(0,y)
void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _outputs.size());
    this->_outputs.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _inputs.size());
    return _inputs.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _inputs.size());
    return _inputs.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _outputs.size());
    return _outputs.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _outputs.size());
    return _outputs.at(idx);
}

void Layer::to_cuda() {
    for(auto& input : _inputs) {
        if(!input.is_empty()) {
            input.to_cuda(_cuda_config ? _cuda_config->stream : nullptr);
        }
    }
    for(auto& output : _outputs) {
        if(!output.is_empty()) {
            output.to_cuda(_cuda_config ? _cuda_config->stream : nullptr);
        }
    }
}


void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
    if(!config) {
        return;
    }
    this->_cuda_config = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const {
    return this->_cuda_config;
}

size_t Layer::input_size() const { return _inputs.size(); }

size_t Layer::output_size() const { return _outputs.size(); }

void Layer::reset_input_size(size_t size) { _inputs.resize(size); }

void Layer::reset_output_size(size_t size) { _outputs.resize(size); }

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_output(0, output1);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_input(1, input2);

    this->set_output(0, output1);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);

    this->set_output(0, output1);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);

    this->set_output(0, output1);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);
    this->set_input(4, input5);

    this->set_output(0, output1);
    return this->forward();
}


//LayerParam
LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, std::string layer_name="")
    :Layer(device_type, layer_type, std::move(layer_name)) {}
   
size_t LayerParam::weight_size() const {
    return _weights.size();
}

void LayerParam::reset_weight_size(size_t size) {
    _weights.resize(size);
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _weights.size());
    return _weights.at(idx);
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _weights.size());
    return _weights.at(idx);
}

//用已经有的weight进行设置
base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _weights.size());
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
    if(!weight.is_empty()) {
        CHECK(weight.device_type() == _device_type);
    }
    _weights[idx] = weight;
    return base::error::Success();
}

//涉及到量化 未实现
base::Status LayerParam::set_weight(int32_t idx, std::vector<int32_t>& dims, 
                                    const void* weight_ptr, base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _weights.size());
    CHECK_NE(weight_ptr, nullptr);

    size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());

    //创建Buffer并设置device_type
    //这里的buffer只是一个内存缓冲，不关心数据类型，只关注底层数据大小和机器类型
    //所以无论是量化还是非量化都可以使用这块buffer
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
    if(device_type != base::DeviceType::kDeviceUnknown) {
        buffer->set_device_type(device_type);
    }

    if(!is_quant_layer) {
        //非量化分支
        tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        _weights.at(idx) = weight;
    } else {
        //量化分支
        tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        _weights.at(idx) = weight;

        const int32_t weight_size = static_cast<int32_t>(weight.size());
        CHECK(weight_size % group_size == 0);
        int32_t scale_num = weight_size / group_size;

        //scales存缩放因子
        //从weight_ptr开始跳过weight参数，取scales参数,weight_ptr是void*指针，所以用reinterpred_cast转换
        _scales = tensor::Tensor{base::DataType::kDataTypeFp32, scale_num, false, nullptr,
                                reinterpret_cast<float*>(reinterpret_cast<int8_t*>(const_cast<void*>(weight_ptr)) + weight_size)};
        
        _scales.set_device_type(device_type);
    }

    return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
    CHECK(!scales.is_empty());
    this->_scales = scales;
}

void LayerParam::set_group_size(int32_t group_size) {
    this->_group_size = group_size;
}

int32_t LayerParam::get_scale_num() const {
    CHECK(!_scales.is_empty()) {
        return static_cast<int32_t>(_scales.size());
    }
}

void LayerParam::to_cuda() {
    //先掉一下Layer
    Layer::to_cuda();
    for(auto& weight : _weights) {
        if(!weight.is_empty()) {
            weight.to_cuda(_cuda_config ? _cuda_config->stream : nullptr);
        }
    }
    if(!scales.is_empty()) {
        scales.to_cuda(_cuda_config ? _cuda_config->stream : nullptr);
    }
}
        


}   //namespace op