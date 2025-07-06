#pragma once
#include "layer.h"

namespace op  {

class RmsNormLayer : public LayerParam {
public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);
    
    base::Status forward() override;
    base::Status check() const override;
private:
    //RmsNorm与Layernorm相似 都是对最后一个维度(一般是特征维度)做归一化处理 所以需要知道最后一个维度大小
    int32_t _dim = 0;  
};

}   //namespace op