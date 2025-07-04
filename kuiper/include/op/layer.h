#pragma once
#include "base/base.h"
#include <string>
#include <vector>

namespace op {

class Layer;    //前向声明Layer
enum class LayerType : uint8_t {
    kLayerUnknown = 0,
    kLayerLinear = 1,
    kLayerEncode = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm = 4,
    kLayerMatmul = 5,
    kLayerRoPe = 6,
    kLayerMHA = 7,
    kLayerSoftmax = 8,
    kLayerAdd = 9,
    kLayerSwiGLU = 10,
};

//算子基类
class BaseLayer {
public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name = "");

    base::DataType data_type() const;
    LayerType layer_type() const;

    virtual base::Status init() = 0;
    
    virtual base::Status forward() = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;
     virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;


    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;
    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    //既能安全地“只读”访问成员，也能在需要时“可写”地修改成员。
    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;
    virtual tensor::Tensor& get_output(int32_t idx) = 0;
    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

    virtual base::Status check() const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight); //使用封装好的张量设置权重
    virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, 
                                    base::DeviceType device_type = base::DeviceType::kDeviceUnknown);  //使用原始数据指针和维度设置权重
                                  
                                    
    const std::string& get_layer_name() const;
    void set_layer_name(const std::string& layer_name);

    base::DeviceType device_type() const;
    void set_device_type(base::DeviceType device_type);


protected:
    std::string _layer_name; //层名
    LayerType _layer_type = LayerType::kLayerUnknown;   //层类型 
    base::DataType _data_type = base::DataType::kDataTypeUnknown; // 层数据类型
    base::DeviceType _device_type = base::DeviceType::kDeviceUnknown;
};

//不带参数的算子派生类
class Layer : public BaseLayer {
public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "")

    base::Status init() override;
    base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const;
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type, ...) const;
    base::Status check() override;

    base::Status forward() override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;


    //传入输入输出
    void set_input(int32_t idx, const tensor::Tensor& input) override;
    void set_output(int32_t idx, const tensor::Tensor& output) override;
    
    //获取输入输出
    const tensor::Tensor& get_input(int32_t idx) const override;
    const tensor::Tensor& get_output(int32_t idx) const override;
    tensor::Tensor& get_input(int32_t idx) override;
    tensor::Tensor& get_output(int32_t idx) override;

    //获取输入输出大小
    size_t input_size() const override;
    size_t output_size() const override;

    //重置输入输出
    void reset_input_size(size_t size);
    void reset_output_size(size_t size);

    virtual void to_cuda();

protected:
    std::vector<tensor::Tensor> _inputs;  // 存放输入的数组
    std::vector<tensor::Tensor> _outputs; // 存放输出的数组
};

class LayerParam : public Layer {
public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type, std::string layer_name="");

    size_t weight_size() const;

    void reset_weight_size(size_t size);
    
    tensor::Tensor& get_weight(int32_t idx);
    const tensor::Tensor& get_weight(int32_t idx) const;

    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;
    base::Status set_weight(int32_t idx, std::vector<int32_t>& dims, const void* weight_ptr,
                            base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;


private:
    std::vector<tensor::Tensor> _weights;   //用于存放额外的权重
};

}   //namespace op