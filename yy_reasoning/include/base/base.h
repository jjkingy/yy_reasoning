#pragma once
#include <cstdint>
#include <string>
#include <glog/logging.h>

//避免未使用变量（unused variable）警告
#define UNUSED(expr)    \ 
    do {                \
        (void)(expr);   \
    } while(0)

namespace model {
enum class ModelBufferType {
    kInputTokens = 0,
    kInputEmbeddings = 1,
    kOutputRMSNorm = 2,
    kKeyCache = 3,
    kValueCache = 4,
    kQuery = 5,
    kInputPos = 6,
    kScoreStorage = 7,
    kOutputMHA = 8,
    kAttnOutput = 9,
    kW1Output = 10,
    kW2Output = 11,
    kW3Output = 12,
    kFFNRMSNorm = 13,
    kForwardOutput = 15,
    kForwardOutputCPU = 16,

    kSinCache = 17,
    kCosCache = 18,
};

}   //namespace model

namespace base {
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

class NoCopyable {
protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};


inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    } else {
        return 0;
    }
}

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

//错误枚举码 不使用enum class, 允许后面的隐式转换
enum StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};

enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
};

//status封装错误信息
class Status {
public:
    Status(int code = StatusCode::kSuccess, std::string message = "");

    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;

    //赋值新的statuscode
    Status& operator=(int code);

    //重载等于/不等于判断条件
    bool operator==(int code) const;
    bool operator!=(int code) const;

    operator int() const;
    operator bool() const;

    // 6. 访问器
    int32_t get_err_code() const;
    const std::string& get_err_msg()  const;

    // 7. 修改错误消息
    void set_err_msg(const std::string& err_msg);

private:
    std::string _message;
    int _code = StatusCode::kSuccess;
};


/*
namespace error 的作用是为常见的错误类型提供统一、易用的工厂函数
集中管理所有常见错误类型，方便维护和拓展
*/
namespace error {
Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}   //namespace base

