#ifndef YY_REASONING_INCLUDE_MHA_H
#define YY_REASONING_INCLUDE_MHA_H
#include <base/cuda_config.h>
#include "layer.h"

namespace op {
class MultiHeadAttention : public op::Layer {
public:
    explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                int32_t head_num. int32_t head_size);
    
    base::Status check() const override;

    void set_pos(int32_t pos);
    void set_layer_idx(int32_t layer_idx);

    base::Status forward() override;

private:
    int32_t _layer_index = 0;
    int32_t _pos = 0;
    int32_t _kv_mul = 0;
    int32_t _kv_dim = 0;
    int32_t _seq_len = 0;
    int32_t _head_num = 0;
    int32_t _head_size = 0;
};

}   // op

#endif