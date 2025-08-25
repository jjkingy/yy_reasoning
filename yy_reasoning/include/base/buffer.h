#ifndef YY_REASONING_INCLUDE_BASE_BUFFER_H_
#define YY_REASONING_INCLUDE_BASE_BUFFER_H_
#include <memory>
// #include "base.h"
#include "base/alloc.h"

namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
private:
    size_t _byte_size = 0;
    void* _ptr = nullptr;
    bool _use_external = false; //是否拥有这块内存的所有权 是否使用的是外部内存
    DeviceType _device_type = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> _allocator;

public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr, 
                    void* ptr = nullptr, bool use_external = false);
    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer& buffer) const;

    void copy_from(const Buffer* buffer) const;

    void* ptr();
    const void* ptr() const;

    DeviceType device_type();

    std::shared_ptr<DeviceAllocator> allocator() const;

    size_t byte_size() const;

    void set_device_type(DeviceType device_type);
};

}   //namespcae base

#endif