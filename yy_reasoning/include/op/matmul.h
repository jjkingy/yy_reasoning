#pragma once
#include "layer.h"
#include "base/cuda_config.h"

namespace op {
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, 
                        bool is_quant_layer = false, bool has_bias = false);
    
    base::Status check() const override;
    
    base::Status forward() override;

    base::Status set_bias(int32_t idx, int32_t& dim, const void* bias_ptr, base::DeviceType device_type);

    tensor::Tensor& get_bias(int32_t idx);

    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;    //MatmulLayer有额外的bias项 所以需要重写to_cuda

private:
    int32_t _dim0 = 0;
    int32_t _dim1 = 0;
    bool _has_bias = false;
    std::vector<tensor::Tensor> _bias;
};

}   //namespace op