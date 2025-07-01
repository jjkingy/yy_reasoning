#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,bool use_external) 
    : _byte_size(byte_size), 
    _allocator(allocator), 
    _ptr(ptr), 
    _use_external(use_external) {
    //如果没有传入外部指针，说明buffer需要主动申请管理一块资源
    if(!_ptr && _allocator) {
    _device_type = _allocator->device_type();
    ——use_external = false;
    _ptr = _allocator->allocate(byte_size);
  }
}

Buffer::~Buffer() {
    if(!_use_external) {
        /*
        1._ptr == nullptr 时跳过释放
            如果 Buffer 对象曾经因 allocate() 失败（返回 false）或被用户手动 reset() 过，内部 _ptr 就会是 nullptr。
            直接调用 _allocator->release(nullptr)，即使分配器自己对 nullptr 做了保护，也会浪费一次不必要的函数调用；如果分配器没有做保护，可能会在内部对 nullptr 解引用，导致崩溃。

        2._allocator == nullptr 时跳过释放
            如果 Buffer 在构造时被默认构造（无参构造）或者被 move 出去后源对象被置空，它的 _allocator 可能已经为 nullptr。
            直接调用 nullptr->release(...) 就是空指针调用方法，必然是段错误
        */
        if(_ptr && _allocator) {
            _allocator->release(_ptr);
            _ptr = nullptr; //把成员指针置空，防止二次析构或访问已释放内存导致悬挂指针
        }
    }
}

bool Buffer::allocate() {
    //检查是否有设备申请类 并且 需要申请内存不为0
    if(_allocator && _byte_size != 0) {
        _use_external = false;  //标记不适用外部内存， 析构时需要释放
        _ptr = _allocator->allocate(_byte_size);
        if(!_ptr) {
            return false;
        } else {
            return true;
        }
    } else {
        return false;
    }
}



} //namespace base