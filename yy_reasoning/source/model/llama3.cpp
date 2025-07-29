#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
//自己写的头文件用<> 强调这是公用api 和标准库头文件一样
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>

namespace model {

void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if(_add_layer) {
    _add_layer->set_cuda_config(config);
    _add_layer->to_cuda();
    }

    if(_rope_layer) {
        _rope_layer->set_cuda_config(config);
        _rope_layer->to_cuda();
    }

    if(_swiglu_layer) {
        _swiglu_layer->set_cuda_config(config);
        _swiglu_layer->to_cuda();
    }

    if(_cls_layer) {
        _cls_layer->set_cuda_config(config);
        _cls_layer->to_cuda();
    }

    if(_embedding_layer) {
        _embedding_layer->set_cuda_config(config);
        _embedding_layer->to_cuda();
    }

    if(_mha_layer) {
        _mha_layer->set_cuda_config(config);
        _mha_layer->to_cuda();
    }

    for (auto& weight_layer : _wq_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wk_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wv_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wo_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w1_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w2_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w3_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& rms_norm_layer : _rmsnorm_layers) {
        if(rms_norm_layer) {
        rms_norm_layer->to_cuda();
        rms_norm_layer->set_cuda_config(config);
        }
    }
}

//未完成
//参数检查 → 设备初始化 → 读权重 → 分配内存 → 预计算 RoPE → 生成采样器
base::Status LLama2Model::init(base::DeviceType device_type) {  //init模型时传入device_type, 决定init在哪个设备上
    // using namespace base;
    //参数检查 token和设备合法性
    if(_token_path.empty()) {
        return base::error::PathNotValid(_token_path);
    }
    if(device == base::DeviceType::kDeviceCPU && _is_quant_model) {
        return base::error::InternalError("The cpu don't support quant model.");
    }

    //设备初始化
    _device_type = device_type;
    if(device_type == base::DeviceType::kDeviceCUDA) {
        cuadSetDevcie(0);   //设置GPU
        _cuda_config = std::make_shared<kernel::CudaConfig>();  //把GPU句柄放在cudaconfig一起管理
        cudaStreamCreate(&_cuda_config->stream);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            return base::error::InternalError("The cuda hanle create failed.");
        }
    }

    //读权重
    //确定模型结构 & 加载权重 & 创建网络层
    base::Status read_status = gen_model_from_file();
    if(!read_status) {
        return read_status;
    }

    //使用已知结构分配推理缓存（中间张量、KV Cache、Pos Embedding Cache）
    init_mem();

    //预计算ROPE
    if(_device_type == base::DeviceType::kDeviceCPU) {

    }

    //生成采样器

    return base::error::Success();
}

//创建llama_layer->创建带参数层->创建不带参数的层->检查每个层是否为空
base::Status LLama2Model::create_layers() {
    using namespace base;

    if(!_llama_layers) {
        _llama_layers = std::make_unique<LLama2Layers>();
    }

    if(!is_quant_model) {
        create_param_layers();
    } else{
        create_param_quant_layers();
    }

    create_nonparam_layers();

    //检查每个层是否创建成功 模型配置是否正确
    if(!llama_layers->_embedding_layer) {
        return error::InternalError("Create embedding layer for llama model failed!");
    }

    //LLaMA2 每层 Transformer Block 通常有两个 RMSNorm（一个在 Attention 前，一个在 FFN 前）
    //再加上输入或输出的 RMSNorm
    if(_llama_layers->_rmsnorm_layers.size() != 2 * _config->_layer_num + 1) {
        return error::InternalError("Create the rmsnorm layers for llama model failed");
    }

    //检查Attention层
    if(_llama_layers->_wq_layers.size() != _config->_layer_num ||
        _llama_layers->_wk_layers.size() != config->_layer_num ||
        _llama_layers->_wv_layers.size() != config->_layer_num ||
        _llama_layers->_wo_layers.size() != config->_layer_num) {
        return error::InternalError(
            "Create the matmul layer in the attenrion and ffn attention layers"
            "for the llama model failed!");
        }
        
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        if(!_llama_layers->_wq_layers.at(i) || !_llama_layers->_wk_layers.at(i) ||
            !_llama_layers->_wv_layers.at(i) || !_llama_layers->_wo_layers.at(i)) {
            return error::InternalError(
                "Create the matmul layer in atten and ffn atten layers"
                "for the llama model failed"
            );
        }
    }

    //检查Feedforward
    if(_llama_layers->_w1_layers.size() != _config->_layer_num ||
       _llama_layers->_w2_layers.size() != _config->_layer_num ||
       _llama_layers->_w3_layers.size() != _config->_layer_num ) {
        return error::InternalError(
            "Create the matmul layer in the feedforward layers for the llama model "
            "failed."
            );
       }

    for(int32_t i = 0; i < _config->_layer_num; i++) {
        if(!_llama_layers->_w1_layers.at(i) || !_llama_layers->_w2_layers.at(i) ||
            !_llama_layers->_w3_layers.at(i)) {
            return error::InternalError(
                "Create the matmul layer in the feedforward layers for the llama model "
                "failed."
                );
        }
    }

    if(!_llama_layers->_rope_layer) {
        return error::InternalError("Create the rope layer for the llama model failed!");
    }

    if(!_llama_layers->_add_layer) {
        return error::InternalError("Create the add layer for the llama model failed!");
    }

    if(!_llama_layers->_mha_layer) {
        return error::InternalError("Create the mha layer for the llama model failed!");
    }

    if(!_llama_layers->_swiglu_layer) {
        return error::InternalError("Create the SwiGLU layer for the llama model failed!");
    }

    return error::Success();
}

void LLama2Model::create_nonparam_layers() {
    CHECK(_llama_layers != nullptr);
    _llama_layers->rope_layer= std::make_shared<op::RoPELayer>(
        _device_type, _config->_dim, _config->_kv_dim, _config->_head_size);

    _llama_layers->_mha_layer= std::make_shared<op::MultiHeadAttention>(
        _device_type, 0, _config->_kv_mul, _config->_kv_dim, _config->_seq_len, _config->_head_num,
        _config->_head_size);

    _llama_layers->_add_layer= std::make_shared<op::VecAddLayer>(_device_type);

    _llama_layers->_swiglu_layer=
        std::make_shared<op::SwiGLULayer>(_device_type, _config->_hidden_dim);

}

void LLama2Model::create_param_quant_layers() {
    CHECK(_is_quant_model);
    CHECK(_llama_layers != nullptr);

    size_t pos = 0;
    int32_t dim = _config->_dim;
    auto cpu_device_type = base::DeviceType::kDeviceCPU;

    //量化模型先处理量化层 未见布局与非量化模型文件不同
    
    //query
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wq = std::make_shared<op::MatmulLayer>(_device_type, dim, dim, true);
        wq->set_weight(0, {dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        wq->set_group_size(_group_size);
        _llama_layers->_wq_layers.push_back(wq);
        pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
    }

    //key
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wk = std::make_shared<op::MatmulLayer>(_device_type, _config->_kv_dim, dim, true);
        wk->set_weight(0, {_config->_kv_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        wk->set_group_size(_group_size);
        _llama_layers->_wk_layers.push_back(wk);
        pos = pos + dim * _config->_kv_dim + wk->get_scale_num() * sizeof(float);
    }

    //value
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wv = std::make_shared<op::MatmulLayer>(_device_type, _config->_kv_dim, dim, true);
        wv->set_weight(0, {_config->_kv_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        wv->set_group_size(_group_size);
        _llama_layers->_wv_layers.push_back(wv);
        pos = pos + dim * _config->_kv_dim + wk->get_scale_num() * sizeof(float);
    }

    //output
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wo = std::make_shared<op::MatmulLayer>(_device_type, dim, dim, true);
        wo->set_weight(0, {dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        wo->set_group_size(_group_size);
        _llama_layers->_wo_layers.push_back(wo);
        pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
    }

    int32_t hidden_dim = _config->_hidden_dim;
    //w1 layers
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w1 = std::make_shared<op::MatmulLayer>(_device_type, hidden_dim, dim, true);
        w1->set_weight(0, {hidden_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        w1->set_group_size(_group_size);
        _llama_layers->_w1_layers.push_back(w1);
        pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
    }
    //w2 layers
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w2 = std::make_shared<op::MatmulLayer>(_device_type, dim, hidden_dim, true);
        w2->set_weight(0, {dim, hidden_dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        w2->set_group_size(_group_size);
        _llama_layers->_w2_layers.push_back(w2);
        pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
    }
    //w3 layers
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w3 = std::make_shared<op::MatmulLayer>(_device_type, hidden_dim, dim, true);
        w3->set_weight(0, {hidden_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        w3->set_group_size(_group_size);
        _llama_layers->_w3_layers.push_back(w3);
        pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
    }

    //wcls layer
    auto cls_layer = std::make_shared<op::MatmulLayer>(_device_type, _config->_vocab_size, dim, true);
    cls_layer->set_group_size(_group_size);
    if(_config->_is_shared_weight) {
        cls_layer->set_weight(0, {_config->_vocab_size, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
    } else {
        cls_layer->set_weight(0, {_config->_vocab_size, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        pos = pos + _config->_vocab_size * dim + cls_layer->get_scale_num() * sizeof(float);
    }
    _llama_layers->_cls_layer = cls_layer;

    //embedding layer
    float* weight_ptr = (float*)_raw_model_data->weight(pos);
    _llama_layers->_embedding_layer = std::make_shared<op::EmbeddingLayer>(
        _device_type, _config->_dim, _config->_seq_len, std::abs(_config->_vocab_size));
    _llama_layers->_embedding_layer->set_weight(0, {std::abs(_config->_vocab_size), dim}, weight_ptr, cpu_device_type);
    weight_ptr += _config->_vocab_size * dim;

    //rmsnorm attenition ffn final
    for(int32_t i = 0; i < 2 * _config->_layer_num + 1; i++) {
        auto rmsnorm_layer = std::make_shared<op::RmsNormLayer>(_device_type, dim);
        rmsnorm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
        _llama_layers->_rmsnorm_layers.push_back(rmsnorm_layer);
        weight_ptr += dim;
    }

}

void LLama2Model::create_param_layers() {
    //合法性检查
    CHECK(!_is_quant_model);
    CHECK(_llama_layers != nullptr);
    // =============================
    // 1. 创建词嵌入（Embedding Layer）
    auto cpu_device_type = base::DeviceType::kDeviceCPU;    //设置设备类型，后面复用
    _llama_layers->_embedding_layer = std::make_shared<op::EmbeddingLayer>(
        _device_type, _config->_dim, _config->_seq_len, std::abs(_config->_vocab_size));
    
    const void* weight_embedding = _raw_model_data->weight(0);
    _llama_layers->_embedding_layer->set_weight(0, {std::abs(_config->_vocab_size), _config->_dim},
                                                weight_embedding, cpu_device_type);

    // =============================
    //2. 创建 Attention 层所需的 Matmul 权重
    int32_t dim = _config->_dim;
    size_t pos = dim * std::abs(_config->_vocab_size) + dim * _config->_layer_num;   //预留每一层RMSnorm的权重

    //Query权重
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wq = std::make_shared<op::MatmulLayer>(_device_type, dim, dim);
        wq->set_weight(0, {dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        _llama_layers->_wq_layers.push_back(wq);
        pos += dim * dim;
    }

    //Key权重
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wk = std::make_shared<op::MatmulLayer>(_device_type, _config->_kv_dim, dim);
        wk->set_weight(0, {_config->_kv_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        pos += _config->_kv_dim * dim;
    }

    //Value权重
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wv = std::make_shared<op::MatmulLayer>(_device_type, _config->_kv_dim, dim);
        wv->set_weight(0, {_config->_kv_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        pos += _config->_kv_dim * dim;
    }

    //Attention输出投影权重
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto wo = std::make_shared<op::MatmulLayer>(_device_type, dim, dim);
        wo->set_weight(0, {dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        _llama_layers->_wo_layers.push_back(wo);
        pos += dim * dim;
    }

    //跳过FFN的RMSNorm
    pos += _config->_layer_num * dim;

    // =============================
    // 3.创建FFN权重
    int32_t hidden_dim = _config->_hidden_dim;

    // FFN w1权重
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w1 = std::make_shared<op::MatmulLayer>(_device_type, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        _llama_layers->_w1_layers.push_back(w1);
        pos += dim * hidden_dim;
    }

    // FFN w2
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w2 = std::make_shared<op::MatmulLayer>(_device_type, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        _llama_layers->_w2_layers.push_back(w1);
        pos += dim * hidden_dim;
    }

    // FFN w3
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto w3 = std::make_shared<op::MatmulLayer>(_device_type, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
        _llama_layers->_w3_layers.push_back(w3);
        pos += dim * hidden_dim;
    }

    //跳过最终RMSNorm 和sin/cos位置编码
    pos += dim;
    pos += _config->_seq_len * _config->_head_size;
    // =============================
    // 4. 分类器层 (输出投影层)
    _llama_layers->_cls_layer = std::make_shared<op::MatmulLayer>(_device_type, _config->vocab_size, dim);

    if(_config->_is_shared_weight) {
        //直接使用embedding权重
        _llama_layers->_cls_layers->set_weight(0, {_config->_vocab_size, dim}, this->_raw_model_data->weight(0), cpu_device_type);
    } else {
        _llama_layers->_cls_layers->set_weight(0, {_config->_vocab_size, dim}, this->_raw_model_data->weight(pos), cpu_device_type);
    }

    // =============================
    // 5. 创建 RMSNorm 层
    size_t rmsnorm_pos = _config->_dim * std::abs(_config->_vocab_size);

    //mutil-head 之前的rmsnorm
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(_device_type, _config->_dim);
        const void* weight_rmsnorm = _raw_model_data->weight(rmsnorm_pos);
        rms_norm_layer->set_weight(0, {_config->_dim}, weight_rmsnorm, cpu_device_type);
        _llama_layers->_rmsnorm_layers.push_back(rms_norm_layer);
        rmsnorm_pos += _config->_dim;
    }
    
    //跳过 attention wq wk wv wo 权重
    rmsnorm_pos += _config->_layer_num * _config->_dim * _config->_dim;
    rmsnorm_pos += _config->_layer_num * _config->_dim * (_config->_kv_head_num * _config->_head_size);
    rmsnorm_pos += _config->_layer_num * _config->_dim * (_config->_kv_head_num * _config->_head_size);
    rmsnorm_pos += _config->_layer_num * _config->_dim * _config->_dim;

    //FFN 前rmsnorm
    for(int32_t i = 0; i < _config->_layer_num; i++) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(_device_type, _config->_dim);
        const void* weight_rmsnorm = _raw_model_data->weight(rmsnorm_pos);
        rms_norm_layer->set_weight(0, {_config->_dim}, weight_rmsnorm, cpu_device_type);
        _llama_layers->_rmsnorm_layers.push_back(rms_norm_layer);
        rmsnorm_pos += _config->_dim;
    }

    //跳过FFN w1 w2 w3
    rmsnorm_pos += _config->_layer_num * _config->_hidden_dim * _config->_dim;
    rmsnorm_pos += _config->_layer_num * _config->_hidden_dim * _config->_dim;
    rmsnorm_pos += _config->_layer_num * _config->_hidden_dim * _config->_dim;

    //最后一层rmsnorm
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(_device_type, _config->_dim);
    const void* weight_rmsnorm = _raw_model_data->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {_config->_dim}, weight_rmsnorm, cpu_device_type);
    _llama_layers->_rmsnorm_layers.push_back(rms_norm_layer);
}

//未完成
void LLama2Model::init_mem() {

    //设置内存分配器
    if(_device_type == base::DeviceType::kDeviceCPU) {
        auto alloc = base::AllocatorFactory<base::CPUDeviceAllocator>::get_instance();
    } else {
        auto alloc = base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();
    }

    //如果是cuda就转移到gpu上
    if(_device_type = base::DeviceType::kDeviceCUDA) {
        CHECK_NE(_cuda_config, nullptr);
        _llama_layers->to_cuda(_cuda_config);
    }

    auto alloc_cpu = base::AllocatorFactory<base::CPUDeviceAllocator>::get_instance();
    auto alloc_cu = base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();

    //未完成， 分配各层Tensor并注册insert到buffer

    

}

}   //namespace model