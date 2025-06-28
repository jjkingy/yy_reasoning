#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"

namespace base {

/* 
1. **资源管理清晰**  
   - NoCopybal禁止拷贝构造函数和赋值构造函数 保证对内存的唯一权
   - `Buffer` 里握着原始指针 `void* ptr_`。为了避免忘记释放，需要保证对象只能“唯一拥有”这块内存，所以 **禁止拷贝**，用移动或 `shared_ptr` 托管。
2. **回调/异步场景安全拿自身**  
   - 推理框架经常把 `Buffer` 塞进 lambda 或线程池。  
   - 在成员函数内部调用 `shared_from_this()` 可以**延长自身生命周期**，防止任务尚未完成对象就被销毁。
3. 对allocator做一层封装，使得满足RAII规范，对allocator得到的内存资源进行管理
   */
class Buffer : public NoCopybal, std::enable_shared_from_this<Buffer> {
private:   
    size_t _byte_size = 0;
    void* _ptr = nullptr;
    bool _use_external = false;
    DeviceType _device_type = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> _allocator;

public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    virtual ~Buffer();

    bool allocate();

};

}

#endif //KUIPER_INCLUDE_BASE_BUFFER_H_