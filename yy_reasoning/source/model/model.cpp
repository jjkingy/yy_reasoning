#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace model {

//打开模型权重文件，读取模型结构配置，做内存映射，设置指向权重数据的指针。
base::Status Model::read_model_file() {
    using namespace base;

    //1检查路径
    if(_model_path.empty()) {
        return error::PathNotValid("Fail to open weight file, the model path is empty!");
    }

    //2打开文件
    //string.data()可以兼容C风格字符串 open打开给mmap用
    int32_t fd = open(_model_path.data(), O_RDONLY);
    if(fd == -1) {
        return error::PathNotValid("Failed to open the weight file " + _model_path +
                               " may be the path does not exist!");
    }
    FILE* file = fopen(_model_path.data(), "rb");
    if (!file) {
        return error::PathNotValid("Failed to open the file. The path may be invalid.");
    }

    //3fread模型配置结构体
    auto config = ModelConfig{};
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
        return error::ModelParseError(
            "Failed to retrieve the configuration information from the model "
            "file.");
    }

    //4如果是量化模型 多读入_group_size参数
    if(_is_quant_model) {
        if(fread(&_group_size, sizeof(int32_t), 1, file) != 1) {
            return error::ModelParseError(
                "Failed to retrieve the group size information from the model file"
            );
        }
    }
    fclose(file);
    //5调用generate_model_infos
    auto gen_status = generate_model_infos(config);
    if (!gen_status) {
        return gen_status;
    }

    //6创建 RawModelDataFp32 或 RawModelDataInt8
    if(!_is_quant_model) {
        _raw_model_data = std::make_shared<RawModelDataFp32>();
    } else {
        _raw_model_data = std::make_shared<RawModelDataInt8>();
    }


    //7fstat 获取文件大小 fstat获取文件身份证 把文件大小权限等多重信息存入stat结构体
    struct stat sb;
    if(fstat(fd, &sb) == -1) {
        close(fd);
        return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
    }
    _raw_model_data->file_size = sb.st_size;

    //8mmap
    _raw_model_data->fd = fd;   //mmap参数(从哪开始、多长、什么权限、怎样共享、用哪个文件、从文件哪里开始)
    _raw_model_data->data = mmap(nullptr, _raw_model_data->file_size, PROT_READ, MAP_PRIVATE, _raw_model_data->fd, 0);
    if(_raw_model_data->data == MAP_FAILED || _raw_model_data->data == nullptr) {
        return error::ModelParseError("Falied to map the weight file" + _model_path + "into momery");
    }

    //9指针偏移赋值
    //这里的int8_t代表的是一个字节为单位偏移 和前面的类没关系
    if(!_is_quant_model) {
        _raw_model_data->weight_data = static_cast<int8_t*>(_raw_model_data->data) + sizeof(ModelConfig);
    } else {
        _raw_model_data->weight_data = static_cast<int8_t*>(_raw_model_data->data) + sizeof(ModelConfig) + sizeof(_group_size);
    }
    if(raw_model_data_->weight_data == nullptr) {
        LOG(ERROR);
        return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                    " into memory, the pointer to weight start address is null");
    }
    return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
    using namespace base;
    _config->_dim = config.dim;
    _config->_hidden_dim = config.hidden_dim;
    _config->_layer_num = config.layer_num;
    _config->_head_num = config.head_num;
    _config->_kv_head_num = config.kv_head_num;
    _config->_seq_len = config.seq_len;

    _config->_kv_dim = (config.dim * config.kv_head_num) / config.head_num;
    _config->_kv_mul = config.head_num / config.kv_head_num;
    _config->_head_size = config.dim / config.head_num;

    //检查支持qwen3
#if defined(QWEN3_SUPPORT)
    _config->_immediate_dim = config._immediate_dim;
#endif
    _config->_is_shared_weight = config.vocab_size > 0 ? true : false;

    // Qwen tokenizer size and embedding size is mismatched
    // refer: https://github.com/QwenLM/Qwen2.5/issues/29
    // if (std::abs(config.vocab_size) != config_->vocab_size_) {
    //   return base::error::ModelParseError(
    //       "Vocabulary size mismatch between the model file and the token list.");
    // }
    _config->_vocab_size = std::abs(config.vocab_size);
    return base::error::Success();
}
    
base::Status Model::gen_model_from_file() {

    _config = std::make_unique<TransformerConfig>();

    //init sentence piece processor
    auto create_encode_status = create_encode_layer();  //create_encode_layers还没实现
    if(!create_encode_status) {
        LOG(ERROR) << "Create the encode layer failed!";
        return create_encode_status;
    }

    //mmap 读模型文件
    auto mmap_status = read_model_file();
    if(!mmap_status) {
        LOG(ERROR) << "Handle model file " << _model_path << " failed!";
        return mmap_status;
    }
    //创建layers
    auto layer_create_status = create_layers();
    if(!layer_create_status) {
        LOG(ERROR) << "Create layers for the model file " << _model_path << " failed!";
        return layer_create_status;
    }

    return error::Success();
}

}