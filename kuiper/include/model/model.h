#pragma once
#include <op/embedding.h>
#include <map>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

//模型接口
namespace model {
class Model {
public:

    virtual base::Status init(base::DeviceType device_type) = 0;
    
    virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;


protected:
    virtual base::Status read_model_file();

    virtual base::Status gen_model_from_file();

    virtual base::Status generate_model_infos(const ModelConfig& config) const;

private:
    virtual base::Status create_layers() = 0;

    virtual void init_mem() = 0;

protected:
    std::unique_ptr<TransformerConfig> _config; //模型结构配置
    int32_t _group_size = 1;
    bool _is_quant_model = false;

    std::shared_ptr<RawModelData> _raw_model_data;  //mmap权重封装
    std::string _model_path;
    std::string _token_path;
    

    base::DeviceType _device_type = base::DeviceType::kDeviceUnknown;
};


}   //namespace model
