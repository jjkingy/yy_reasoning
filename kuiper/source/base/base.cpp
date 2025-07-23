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
Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }

Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}

//重载运算符<<
std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.get_err_msg();
    return os;
}

}  // namespace error

}   //namespace base