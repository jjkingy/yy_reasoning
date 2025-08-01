#pragma once
#include "base/base.h"
#include "base/buffer.h"
#include <driver_types.h>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>

namespace tensor{

class Tensor {
public:
    explicit Tensor() = default;
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc, void* ptr);
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    void to_cpu();

    void to_cuda(cudaStream_t stream = nullptr);


    bool is_empty() const;
    
    //初始化buffer并allocate
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType, bool need_alloc, void* ptr);

    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc);

    int32_t dims_size() const;

    //分配一个外部buffer 只用buffer 不管数据
    bool assign(std::shared_ptr<base::Buffer> buffer);

    size_t byte_size() const;

    void set_device_type(base::DeviceType device_type) const;

    //对数据张量访问
    template<typename T>
    T* ptr();

    template<typename T>
    const T* ptr() const;

    template<typename T>
    T* ptr(int64_t index);

    template<typename T>
    const T* ptr(int64_t index) const;

    size_t size() const;

    int32_t get_dim(int32_t idx) const;

    std::vector<size_t> strides() const;

    base::DeviceType device_type() const;

    size_t byte_size() const;


private:
    std::shared_ptr<base::Buffer> _buffer;
    size_t _size = 0;
    std::vector<int32_t> _dims;
    base::DataType _data_type = base::DataType::kDataTypeUnknown;
};

template<typename T>
T* Tensor::ptr() {
    if(!_buffer) {
        return nullptr;
    }
    return reinterpret_cast<T*>(_buffer->ptr());
}

template<typename T>
const T* Tensor::ptr() const {
    if(!_buffer) {
        return nullptr;
    }
    return const_cast<const T*>(reinterpret_cast<T*>(_buffer->ptr()));
}

template<typename T>
T* Tensor::ptr(int64_t index) {
    CHECK(_buffer != nullptr && _buffer->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or pointer is empty";
    return reinterpret_cast<T*>(_buffer->ptr()) + index;
}

template<typename T>
const T* Tensor::ptr(int64_t index) const {
    CHECK(_buffer != nullptr && _buffer->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or pointer is empty";
    return const_cast<const T*>(reinterpret_cast<T*>(_buffer->ptr())) + index;
}

}   //namespace tensor

