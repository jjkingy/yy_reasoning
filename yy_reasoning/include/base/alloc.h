// ┌───────────┐        allocate / release        ┌────────────┐
// │  Buffer   │ ───────────────────────────────▶ │ Allocator  │
// │ (RAII)    │ ◀─────────────────────────────── │  (CPU / GPU)│
// └───────────┘        owns ptr & bytes         └────────────┘
#ifndef YY_REASONING_BASE_ALLOC_H_
#define YY_REASONING_BASE_ALLOC_H_

#include "base.h"
#include <memory>
#include <map>


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
                        MemcpyKind memcpy_ind = MemcpyKind::kMemcpyCPU2CPU, 
                        void* stream = nullptr, bool need_sync = false) const;

    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

private:
    DeviceType _device_type = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    //explicit只能显示构造，禁止隐形转换
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;

    // void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const override;
};

struct CudaMemoryBuffer {
    void* _data;
    size_t _byte_size;
    bool _busy;

    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : _data(data), _byte_size(byte_size), _busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
private:
    //大块显存一般用于KV cache 模型参数等长生命周期内存，这种内存长期复用且申请/释放开销大，不适合频繁清理
    mutable std::map<int, size_t> _no_busy_cnt;
    //map<int,vector>因为显卡不止一块，map记录多张显卡上的多个cudaMemoryBuffer
    mutable std::map<int, std::vector<CudaMemoryBuffer>> _big_buffers_map;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> _cuda_buffers_map;
};


// class CPUDeviceAllocatorFactory {
// public:
//     static std::shared_ptr<CPUDeviceAllocator> get_instance() {
//         static auto instance = std::make_shared<CPUDeviceAllocator>();
//         return instance;
//     }
// };

// class CUDADeviceAllocatorFactory {
// public:
//     static std::shared_ptr<CUDADeviceAllocator> get_instance() {
//         static auto instance = std::make_shared<CUDADeviceAllocator>();
//         return instance;
//     }
// };
//使用模板代替上面写法
template<typename Alloc>
class AllocatorFactory {
public:
    //使用C++11局部静态变量保证线程安全
    static std::shared_ptr<Alloc> get_instance() {
        static auto instance = std::make_shared<Alloc>();
        return instance;
    }
};


}   //namespace base

#endif