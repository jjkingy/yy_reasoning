#include "op/encode.h"
#include <glog/logging.h>
#include "base/unicode.h"

namespace op {

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using namespace sentencepiece::util;
    spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto rc = spe->Load(_token_model_path);
    if(rc.code() != StatusCode::kOk) {
        LOG(FATAL)
            << "The token model path is not valid, please check the path and type of token model.";
    }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe != nullptr);

    //sentencepiece
    std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
    if(_has_bos) {
        input_ids.insert(input_ids.begin(), spe->bos_id());
    }
    if(_has_eos) {
        input_ids.push_back(spe->eos_id());
    }
    return input_ids;
}

std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe != nullptr);
    std::vector<int32_t> token_ids{token_id};
    return this->spe->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(spe != nullptr);
    return this->spe->DecodeIds(token_ids);
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    CHECK(this->spe != nullptr);
    return token_id == this->spe->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
    CHECK(spe != nullptr);
    return spe->GetPieceSize();
}


#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;

    //读文件
    std::ifstream f(_token_model_path);
    CHECK(f.is_open())
      << "The token model path is not valid, please check the path and type of token model.";
    
    json data;
    try {
        //解析json文件
        data = json::parse(f);
    } catch (json::parse_error&) {
        LOG(FATAL)
            << "The token model path is not valid, please check the path and type of token model.";
    }

    const auto& datas = data["added_tokens"];

    //ankerl::unordered_dense::map = 高性能、紧凑型 unordered_map
    ankerl::unordered_dense::map<std::string, int> special_tokens;

    //处理特殊单词
    for(const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    //读取常规单词
    //把 JSON 里的 UTF-8 token（人类可读的字符串） → 转换成「字节序列形式的 std::string」
    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& voacbs = data["model"]["vocab"];
    const auto& vocab_items = voacbs.item();
    for(const auto& v : vocab_items) {
        //BPE使用字节级(8bit) vocab只有2的8次方 = 256 并且保证通用性 所有utf-8可编码成字节
        //如果使用字符级 vocab要涵盖所有数字汉字字母太大了
        const auto cpts = unicode_cpt_from_utf8(v.key());
        std::string key;
        for(const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    _bos_id = special_tokens["<|begin_of_text|>"];
    _eos_id = special_tokens["<|end_of_text|>"];
    _stop_token1 = _eos_id;
    _stop_token2 = special_tokens["<|eos_id|>"];

    _num_token = encoder.size() + special_tokens.size();
    _tiktoken = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(this->_tiktoken != nullptr);
    
    //hugging face bpe编码时 例如:"hello, the" the作特殊单词，the第一个位置的空格编码为Ġ
    std::map<std::string, string> replacements;
    replacements[" "] = "Ġ";
    std::string s = absl::StrReplaceAll(sentence, replacements);
    auto input_ids = this->_tiktoken->encode(s);

    if(_has_bos) {
        input_ids.insert(input_ids.begin(), _bos_id);
    }
    if(_has_eos) {
        input_ids.push_back(_eos_id);
    }
    return input_ids;
}

// std::string BpeEncodeLayer::decode(int32_t token_id) const { return ""; }
std::string BpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(this->_tiktoken != nullptr);
    std::vector<int32_t> token_ids{token_id};
    
    auto s = _tiktoken->decode(token_ids);
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;
}

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(this->_tiktoken != nullptr);
    auto s = _tiktoken->decode(token_ids);
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;
}

bool is_sentence_ending(int32_t token_id) const {
    if(token_id == _stop_token1 || token_id == _stop_token2) {
        return true;
    } else {
        return false;
    }
}

int32_t vocab_size() const {
    CHECK(this->_tiktoken != nullptr);
    return _num_token;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path);

    json data = json::parse(f);
    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for(const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();

    for(const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for(const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    _bos_id= special_tokens["<|im_start|>"];
    _eos_id= special_tokens["<|im_end|>"];
    _stop_token1= _eos_id;
    _stop_token2= special_tokens["<|endoftext|>"];

    _num_token= encoder.size() + special_tokens.size();
    _tiktoken= std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

}   //op