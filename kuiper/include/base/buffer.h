#pragma once
#include <memory>
// #include "base.h"
#include "base/alloc.h"

namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr, void* ptr = nullptr, bool use_external = false);
    virtual ~Buffer();

    bool allocate();

    void* ptr();
    const void* ptr() const;

private:
    size_t _byte_size = 0;
    void* ptr = nullptr;
    bool _use_external = false; //是否拥有这块内存的所有权 是否使用的是外部内存
    DeviceType _device_type = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> _allocator;
};

}   //namespcae base
