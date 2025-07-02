#pragma once
#include <cstdint>
#include <string>
#include <glog/logging.h>


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

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

//错误枚举码
enum class StatusCode : uint8_t {
    kSuccess = 0.
    kFunctionUnImplement = 1,
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

}   //namespace base

