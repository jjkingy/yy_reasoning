// ┌───────────┐        allocate / release        ┌────────────┐
// │  Buffer   │ ───────────────────────────────▶ │ Allocator  │
// │ (RAII)    │ ◀─────────────────────────────── │  (CPU / GPU)│
// └───────────┘        owns ptr & bytes         └────────────┘
#pragma once
#include "base.h"
#include <memory>

namespace base{
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : _device_type(device_type) {}

    //const成员函数保证只是'查询'对象状态而不改变对象内容
    virtual DeviceType device_type() const { return _device_type; }

    //设置纯虚函数保证子类必须重写
    virtual void release(void* ptr) const = 0;
    virtual void* allocate(size_t byte_size) const = 0;

    //把memcpy集成起来由父类实现
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t size, 
                        MemcpyKind memcpy_ind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr, bool need_sync = false) const;

    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

private:
    DeviceType _device_type = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;

    // void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
}


}   //namespace base
