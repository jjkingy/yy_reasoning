#pragma once
#include <cstdint>
#include <cstddef>

//模型权重封装模块
// +--------------------+--------------------+-----------------------+
// |  ModelConfig       |  group_size_ (int) |   权重数据 (FP32/INT8) |
// +--------------------+--------------------+-----------------------+
// ^                    ^                    ^
// |                    |                    |
// data              +sizeof()         weight_data
// 封装模型文件的加载信息（比如文件描述符 fd、内存映射后的 data）
// 提供对模型权重的访问接口 weight(offset)，通过偏移量访问模型中存储的某块权重


namespace model {
struct RawModelData {
    ~RawModelData();
    int32_t fd = -1;    //文件描述符
    size_t file_size = 0;   //文件大小
    void* data = nullptr;   //整体文件数据
    void* weight_data = nullptr;    //权重数据

    virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
    const void* weight(size_t offset) const override;
};


struct RawModelDataInt8 : RawModelData {
    const void* weight(size_t offset) const override;
};

}   //namespace model