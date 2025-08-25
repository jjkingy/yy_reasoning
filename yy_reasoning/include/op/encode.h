#ifndef YY_REASONING_INCLUDE_ENCODE_H_
#define YY_REASONING_INCLUDE_ENCODE_H_
#include <sentencepiece_processor.h>
#include "layer.h"

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#endif

namespace op {
class EncodeLayerBase : public Layer {
public:
    explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
        : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
            _has_bos(has_bos),
            _has_eos(has_eos),
            _token_model_path(std::move(token_model_path)) {}

    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    virtual std::string decode(int32_t token_id) const = 0;

    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    virtual bool is_sentence_ending(int32_t token_id) const = 0;

    virtual int32_t vocab_size() const = 0;

protected:
    bool _has_bos = true;
    bool _has_eos = false;
    std::string _token_model_path;
};


class SpeEncodeLayer : public EncodeLayerBase {
public:
    explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
class BpeEncodeLayer : public EncodeLayerBase {
public:
    explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

protected:
    int32_t _bos_id= -1;
    int32_t _eos_id= -1;
    int32_t _stop_token1= -1;
    int32_t _stop_token2= -1;
    int32_t _num_token= 0;
    std::unique_ptr<tiktoken::tiktoken> _tiktoken;
};

class QwenEncodeLayer : public BpeEncodeLayer {
public:
    explicit QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
};

#endif

}   //op