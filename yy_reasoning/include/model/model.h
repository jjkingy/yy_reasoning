#ifndef YY_REASONING_INCLUDE_MODEL_MODEL_H_
#define YY_REASONING_INCLUDE_MODEL_MODEL_H_
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

    explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

    virtual base::Status init(base::DeviceType device_type) = 0;
    
    virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

    virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

    virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

    virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx, 
                                                                    int32_t token_pos) const;


protected:
    virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

    virtual base::Status read_model_file();

    virtual base::Status gen_model_from_file();

    virtual base::Status create_encode_layer();

    virtual base::Status generate_model_infos(const ModelConfig& config) const;

private:
//必须重写的接口
    virtual base::Status create_layers() = 0;

    virtual void init_mem() = 0;

    virtual void create_param_layers() = 0;

    virtual void create_param_quant_layers() = 0;

    virtual void create_nonparam_layers() = 0;

//如果改成 private，子类无法直接访问这些变量，必须提供大量的 get/set 接口。
//这在框架类设计里通常是反模式，因为这个类的主要用途就是让子类重写、扩展。
protected:
    std::unique_ptr<TransformerConfig> _config; //模型结构配置
    int32_t _group_size = 1;
    bool _is_quant_model = false;

    std::shared_ptr<RawModelData> _raw_model_data;  //mmap权重封装
    std::string _model_path;
    std::string _token_path;
    std::unique_ptr<op::EncodeLayerBase> _encode_layer;
    std::unique_ptr<Sampler::Sampler> _sampler;
    std::map<base::ModelBufferType, tensor::Tensor> _buffers;
    
    base::TokenizerType _tokenizer_type = base::TokenizerType::kTokenizerTypeUnknown;
    base::ModelType _model_type = base::ModelType::kModelTypeUnknown;
    base::DeviceType _device_type = base::DeviceType::kDeviceUnknown;
};


}   //namespace model

#endif