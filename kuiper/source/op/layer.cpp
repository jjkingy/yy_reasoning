#include "op/layer.h"
#include <glog/logging.h>
#include <cstdarg>
#include "tensor/tensor.h"


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

void BaseLayer::set_device_type(base::DeviceType device_type) { _device_name = device_type; }

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
   
size_t LayerParam::weights_size() const {
    return _weights.size();
}

void LayerParam::reset_weight_size(size_t size) const {
    _weights.resize(size);
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return _weights.at(idx);
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return _weights.at(idx);
}
base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _weight.size());
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
    if(!weight.is_empty()) {
        CHECK(weight.device_type() == _device_type);
    }
    _weights[i] = weight;
    return base::error::Success();
}

//涉及到量化 未实现
base::Status set_weight(int32_t idx, std::vector<int32_t>& dims, const void* weight_ptr,
                        base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;
        


}   //namespace op