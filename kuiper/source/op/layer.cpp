#include "op/layer.h"
#include <glog/logging.h>


namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name)
    :_device_type(device_type),
    _layer_type(layer_type),
    _data_type(data_type),
    _layer_name(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return _data_type; }

LayerType BaseLayer::layer_type() const { return _layer_type; }

const std::string& BaseLayer::get_layer_name() const {return _layer_name;}

void BaseLayer::set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }

base::DeviceType BaseLayer::device_type() const { return _device_type;}

void BaseLayer::set_device_type(base::DeviceType device_type) { _device_name = device_type; }


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

const tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, _outputs.size());
    return _outputs.at(idx);
}

size_t Layer::input_size() const { return _inputs.size(); }

size_t Layer::output_size() const { return _outputs.size(); }

void Layer::reset_input_size(size_t size) { _inputs.resize(size); }

void Layer::reset_output_size(size_t size) { _outputs.resize(size); }



}   //namespace op