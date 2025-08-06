#pragma once
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace model {

struct LLama2Layers {
    std::shared_ptr<op::Layer> _add_layer;                // 残差连接层（Residual Add）
    std::shared_ptr<op::Layer> _rope_layer;               // Rotary 位置编码（RoPE，用于 Attention 的 Q/K）
    std::shared_ptr<op::Layer> _swiglu_layer;             // SwiGLU 激活层（FeedForward 激活函数）
    std::shared_ptr<op::Layer> _mha_layer;                // 多头注意力封装层（Multi-Head Attention）

    std::vector<std::shared_ptr<op::Layer>> _wq_layers;   // Query 投影层（Wq）
    std::vector<std::shared_ptr<op::Layer>> _wk_layers;   // Key 投影层（Wk）
    std::vector<std::shared_ptr<op::Layer>> _wv_layers;   // Value 投影层（Wv）
    std::vector<std::shared_ptr<op::Layer>> _wo_layers;   // Attention 输出投影层（Wo）

    std::vector<std::shared_ptr<op::Layer>> _w1_layers;   // FeedForward 第一层（W1）
    std::vector<std::shared_ptr<op::Layer>> _w2_layers;   // FeedForward 第二层（W2）
    std::vector<std::shared_ptr<op::Layer>> _rmsnorm_layers; // RMSNorm 层（替代 LayerNorm）
    std::vector<std::shared_ptr<op::Layer>> _w3_layers;   // FeedForward 的辅助投影层（用于 SwiGLU 的分支）

    std::shared_ptr<op::Layer> _cls_layer;                // 分类头（可选，用于下游分类任务）
    std::shared_ptr<op::Layer> _embedding_layer;          // 词嵌入层（Token → Embedding）

    void to_cuda(std::shared_ptr<kernel::CudaConfig> config); // 将所有层转移到 CUDA（GPU）上
};

class LLama2Model : public Model {
public:
    
    base::Status init(base::DeviceType device_type) override;

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor, int& next) const override;

private:
    void init_mem() override;

    base::Status create_layers() override;

    void create_param_layers() override;

    void create_nonparam_layers() override;

    void create_param_quant_layers() override;

    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

private:
    std::shared_ptr<kernel::CudaConfig> _cuda_config;
    //后面如果没有共享就改成Unique_ptr
    std::unique_ptr<model::LLama2Layers> _llama_layers;    //把所有op层组织在一起，方便管理和迁移到cuda
};


}   //namespace model