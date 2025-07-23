#pragma once

namespace model {
struct ModelConfig {
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t layer_num = 0;
    int32_t head_num = 0;
    int32_t kv_head_num = 0;
    int32_t vocab_size = 0;
    int32_t seq_len = 0;
#ifdef QWEN3_SUPPORT
    int32_t _immediate_dim = 0;
#endif
};

struct TransformerConfig {
    int32_t _kv_dim = 0;    //KV 投影后每个 token 的向量长度；等于 _head_size * _kv_head_num。这个值用来给 kv-cache 分配显存。
    int32_t _kv_mul = 0;    //一个快速计算「Q 头数 ÷ KV 头数」的整数倍率，_head_num / _kv_head_num。解码阶段用它来复用 KV。
    int32_t _head_size = 0;
    int32_t _vocab_size = 0;

    int32_t _dim = 0;   //embedding后的维度
    int32_t _hidden_dim = 0;
    int32_t _layer_num = 0;
    int32_t _head_num = 0;  //多头注意力总头数
    int32_t _kv_head_num = 0;  //kv头数
    int32_t _seq_len = 0;
    bool _is_shared_weight = false;
#ifdef QWEN3_SUPPORT
    int32_t _immediate_dim = 0;
#endif
};

}  // namespace model
