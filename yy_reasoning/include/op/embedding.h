#pragma once
#include <utility>
#include "layer.h"
namespace op {
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

    base::Status check() const override;

    base::Status forward() override;

 private:
  int32_t _dim = 0;
  int32_t _seq_len = 0;
  int32_t _vocab_size = 0;
};

}   //namespace op