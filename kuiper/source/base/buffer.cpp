#include"base/buffer.h"
#include<glog/logging.h>

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool use_external)
        :_byte_size(byte_size), _allocator(allocator), _ptr(ptr), _use_external(use_external) {
    //如果外部传入的内存指针为nullptr且分配器不为空，则分配一块自己管理的内存
    if(!_ptr && _allocator) {
        _device_type = _allocator->device_type();
        _use_external = false;
        _ptr = _allocator->allocate(_byte_size);
    }
    
}

Buffer::~Buffer() {
    if(!_use_external) {
        //只有不使用外部内存是才释放内存
        if(_ptr && _allocator) {
            _allocator->release(_ptr);
            _ptr = nullptr; //指针置空
        }
    }
}

//allocate可以手动控制内存分配时机 不一定要在初始化的时候就一定分配好内存
bool Buffer::allocate() {
    if(_allocator && _byte_size != 0) {
        _use_external = false;
        _ptr = _allocator->allocate(_byte_size);
        if(!_ptr) {
            return false;
        }else {
            return true;
        }
    }else {
        return false;
    }
}

}   //namespace base
