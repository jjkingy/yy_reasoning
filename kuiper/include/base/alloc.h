#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base/base.h"
// using namespace base;
namespace base {
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

//使用设备资源申请器类管理资源的申请和释放 为不同的设备对资源申请和释放提供统一的接口
//将设备资源管理器抽象为父类 方便不同的设备继承并重写方法
//allocator只是资源申请类 相当于一个分配器 本身不持有任何内存 
class DeviceAllocator {
public:
    //explicit禁止隐式转换
    explicit DeviceAllocator(DeviceType device_type) : _device_type(device_type) {}

    virtual DeviceType device_type() const {  return _device_type;}
    
    //释放内存
    virtual void release(void* ptr) const = 0;

    //申请内存
    virtual void* allocate(size_t byte_size) const = 0;

    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;

    virtual void memset_zero(void* ptr, size_t byte_size,void* stream, bool need_sync = false);

private:
    DeviceType _device_type = DeviceType::kDeviceUnknown;
};

//CPU设备资源管理器
class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};


//CUDA设备资源管理器
class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
private:

};



}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_