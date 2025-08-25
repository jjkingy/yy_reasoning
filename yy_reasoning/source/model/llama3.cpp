#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
// #include "../op/kernels/cpu/rope_kernel.h"
// #include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
//自己写的头文件用<> 强调这是公用api 和标准库头文件一样

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
    _sampler = std::make_unique<sampler::ArgmaxSampler>(_device_type);

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

/*
相同的算子层复用中间缓存 减少malloc/free cudaMalloc/cudaFree 提升性能(减少内存碎片 减少时间开销)
预分配KV Cache(动态扩展和会导致内存碎片)
集中处理Tensor在不同设备cpu/gpu上的分配
*/
void LLama2Model::init_mem() {
    // 1. 选择分配器
    std::shared_ptr<base::DeviceAllocator> alloc;
    if (_device_type == base::DeviceType::kDeviceCPU) {
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
        alloc = base::CUDADeviceAllocatorFactory::get_instance();
    }

    // 如果使用 GPU，将所有算子权重迁移到 GPU
    if (_device_type == base::DeviceType::kDeviceCUDA) {
        CHECK(_cuda_config != nullptr);
        _llama_layers->to_cuda(_cuda_config);
    }

    // CPU 分配器备用（比如 Pos Tensor、最终结果等需要 Host 可访问的内存）
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    // 2. 输入相关缓存
    tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, _config->_dim, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

    // 3. RoPE sin/cos 缓存
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32,
                             _config->_head_size * _config->_seq_len, true, alloc);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32,
                             _config->_head_size * _config->_seq_len, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

    // 4. RMSNorm 和 FFN 中间输出
    tensor::Tensor rms_output(base::DataType::kDataTypeFp32, _config->_dim, true, alloc);
    tensor::Tensor w1_output(base::DataType::kDataTypeFp32, _config->_hidden_dim, true, alloc);
    tensor::Tensor w3_output(base::DataType::kDataTypeFp32, _config->_hidden_dim, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
    CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
    CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
    CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
    CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
    CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

    // 5. KV Cache (增量推理关键)
    //KV Cache N层 * seq_len(token) * dim
    tensor::Tensor key_cache(base::DataType::kDataTypeFp32,
                             _config->_layer_num, _config->_seq_len,
                             _config->_kv_dim, true, alloc);
    tensor::Tensor value_cache(base::DataType::kDataTypeFp32,
                               _config->_layer_num, _config->_seq_len,
                               _config->_kv_dim, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

    // 6. Query 和 Attention 中间结果
    tensor::Tensor query(base::DataType::kDataTypeFp32, _config->_dim, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kQuery, query));

    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

    tensor::Tensor attn(base::DataType::kDataTypeFp32,
                        _config->_head_num, _config->_seq_len, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
    CHECK(insert_buffer(ModelBufferType::kAttnOutput, query)); // 直接复用 query buffer

    // 7. 模型最终输出 (logits)
    tensor::Tensor forward_output(base::DataType::kDataTypeFp32,
                                  _config->_vocab_size, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));

    // GPU 推理时需要在 CPU 上额外保留一份输出（用于从 GPU 拷贝结果）
    if (_device_type == base::DeviceType::kDeviceCUDA) {
        tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32,
                                          _config->_vocab_size, true, alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
    }
}


op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) {
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
    
    if(input_tokens.size() != tokens.size()) {
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()), _config->_dim});
    }
    for(int32_t i = 0; i < tokens.size(); i++) {
        input_tokens.index<int32_t>(i) = tokens.at(i);
    }

    auto input_token_num = 
        tensor::Tensor(base::DeviceType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
    LOG_IF(FATAL, !_llama_layers->_embedding_layer)
        << "The embedding layer in the llama2 model is null pointer";
    STATUS_CHECK(_llama_layers->_embedding_layer->forward(input_tokens, input_token_num, input_embeddings));

    op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
    return output;
}

//forward调用子函数未完成
base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
    if(input.is_empty()) {
        return base::error::InvalidArgument("The input tensor is empty.");
    }
    if(_device_type == base::DeviceType::kDeviceCPU && is_quant_model) {
        return base::error::InternalError("unsupport int 8 in cpu");
    }

    for(int32_t layer_idx = 0; layer_idx < _config->_layer_num; ++layer_idx) {
        attention_rms(layer_idx, input);
        // attention (wq wk wv @ input)
        attention_qkv(layer_idx, pos_tensor);
        // multi-head attention
        attention_mha(layer_idx, pos_tensor);
        // feed forward
        feed_forward(layer_idx, input);
    }
    cls_logits(input);

    return base::error::Success();
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK(_llama_layers != nullptr);
    
    //attention rmsnorm
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op::Layer> rmsnorm_layer = _llama_layers->_rmsnorm_layers.at(layer_idx);
    if (!rmsnorm_layer) {
        LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
    }
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

//未完成rope后端实现
void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(_llama_layers != nullptr);

    //kv cache
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);
    
    const auto& [key, val] = slice_kv_cache(layer_idx, pos);

    //wq wk wv @ input
    //query
    const auto& query_layer = _llama_layers->_wq_layers.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in attention block is null pointer";
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

    //key
    const auto& key_layer = _llama_layers->_wk_layers.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in attention block is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

    //value
    const auto& value_layer = _llama_layers->_wv_layers.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

    //rope
    CHECK_NE(_llama_layers->_rope_layer, nullptr)
        << "The Rope layer in the attention block is null pointer.";
    STATUS_CHECK(_llama_layers->_rope_layers->forward(
        query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(_llama_layers != nullptr);

    //mha
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

    // VAL = [val1,val2,...val t]
    // output @ VAL = 最终的结果
    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

    const auto& mha_layer = _llama_layers->_mha_layer;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention is null pointer";
    int pos = pos_tensor.index<int32_t>(0);
    //dynamic_pointer_cast 用于多态指针之间的安全转换(只适用shared_ptr)
    //它解决的就是：当你手里只有一个基类智能指针，但你需要用到派生类的接口/数据时，怎么安全转换
    //https://blog.csdn.net/qq_28127741/article/details/120625183
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
    
    //wo @ attention output 最后的linear
    tensor::Tensor atten_output = get_buffer(ModelBufferType::kAttenOutput);
    const auto& wo_layer = _llama_layers->_wo_layers.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output is null pointer."
    STATUS_CHECK(wo_layer->forward(mha_output, atten_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK(_llama_layers != nullptr);

    //residual add
    CHECK_NE(_llama_layers->_add_layer, nullptr) << "The add layer in the feedword block is null pointer";
    STATUS_CHECK(_llama_layers->_add_layer->forward(input, get_buffer(ModelBufferType::kAttenOutput), input));

    //ffn rmsnorm
    tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);  //获取ffn_rmsnorm的中间缓存
    const auto& ffn_rmsnorm = _llama_layers->_rmsnorm_layers.at(_config->_layer_num, layer_idx);
    CHECK_NE(ffn_rmsnorm, nullptr) << "The final rmsnorm layer in feed forward block is null pointer";
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

    // gate wq
    tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = _llama_layers->_w1_layers.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

    //w3 up_proj
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = _llama_layers->_w3_layers.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

    //swiGLU
    CHECK_NE(_llama_layers->_swiglu_layer, nullptr)
        << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(_llama_layers->_swiglu_layer->forward(w1_output, w3_output, w1_output));

    //w2 down_proj
    tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = _llama_layers->_w2_layers.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

    //residual add
    STATUS_CHECK(_llama_layers->_add_layer->forward(input, w2_output, input));

}

}   //namespace model