#include "base/base.h"
#include <string>


namespace base{
Status::Status(int code, std::string message)
    :_code(code), _message(message) {}


//赋值新的statuscode
Status& Status::operator=(int code) {
    _code = code;
    return *this;
}

//重载等于/不等于判断条件
bool Status::operator==(int code) const {
    return _code == code ? true : false;
}

bool Status::operator!=(int code) const {
    return _code != code ? true : false;
}

Status::operator int() const {
    return _code;
}
Status::operator bool() const {
    return _code == StatusCode::kSuccess;
}

int32_t Status::get_err_code() const {
    return _code;
}
const std::string& Status::get_err_msg()  const {
    return _message;
}


void Status::set_err_msg(const std::string& err_msg) {
    _message = err_msg;
}


namespace error {
//使用{}统一初始化列表
Status Success(const std::string& err_msg = "") {
    return Status{StatusCode::kSuccess, err_msg};
}

Status FunctionNotImplement(const std::string& err_msg = "") {
    return Status{StatusCode::kFunctionUnImplement, err_msg};
}

Status InvalidArgument(const std::string& err_msg = ""){
    return Status{StatusCode::kInvalidArgument, err_msg};
}

}   //namespace error

}   //namespace base