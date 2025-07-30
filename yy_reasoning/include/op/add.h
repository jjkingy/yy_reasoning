#pragma once
#include "base/base.h"
#include "layer.h"

namespace op {
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;
    
    base::Status forward() override;
};

}   //namespace op