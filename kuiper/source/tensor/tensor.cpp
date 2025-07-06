include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>


namespace tensor {
static size_t data_type_size(base::DataType data_type) {
    switch(data_type) {
        case base::DataType::kDataTypeFp32:
            return 4;
        case base::DataType::kDataTypeInt8:
            return 1;
        case base::DataType::kDataTypeInt32:
            return 4;
        default:
            LOG(FATAL) << "unknown data type size for " << int(data_type);
            return 0;
    }
}


Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) 
                :_data_type(data_type) {
    _dims.emplace_back(dim0);
    _size = dim0;  
    if(need_alloc && alloc) {   //需要分配并且有分配器
        allocate(alloc);
    } else {    //不需要分配，使用外部传入得内存
        if(ptr != nullptr) {    //外部传入ptr不空且need_alloc必须为false
            CHECK(need_alloc == false)
                << "The need_alloc is ture when ptr is not nullptr."; 
            init_buffer(alloc, _data_type, need_alloc, ptr);
        }
    }
}


Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr)
                :_data_type(data_type) {
    _dims.emplace_back(dim0);
    _dims.emplace_back(dim1);
    _size = dim0 * dim1;  
    if(need_alloc && alloc) {   //需要分配并且有分配器
        allocate(alloc);
    } else {    //不需要分配，使用外部传入得内存
        if(ptr != nullptr) {    //外部传入ptr不空且need_alloc必须为false
            CHECK(need_alloc == false)
                << "The need_alloc is ture when ptr is not nullptr."; 
            init_buffer(alloc, _data_type, need_alloc, ptr);
        }
    }

}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr)
                :_data_type(data_type) {
    _dims.emplace_back(dim0);
    _dims.emplace_back(dim1);
    _dims.emplace_back(dim2);
    _size = dim0 * dim1 * dim2; 
    if(need_alloc && alloc) {   //需要分配并且有分配器
        allocate(alloc);
    } else {    //不需要分配，使用外部传入得内存
        if(ptr != nullptr) {    //外部传入ptr不空且need_alloc必须为false
            CHECK(need_alloc == false)
                << "The need_alloc is ture when ptr is not nullptr."; 
            init_buffer(alloc, _data_type, need_alloc, ptr);
        }
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr)
                :_data_type(data_type) {
    _dims.emplace_back(dim0);
    _dims.emplace_back(dim1);
    _dims.emplace_back(dim2);
    _dims.emplace_back(dim3);
    _size = dim0 * dim1 * dim2 * dim3;
    if(need_alloc && alloc) {   //需要分配并且有分配器
        allocate(alloc);
    } else {    //不需要分配，使用外部传入得内存
        if(ptr != nullptr) {    //外部传入ptr不空且need_alloc必须为false
            CHECK(need_alloc == false)
                << "The need_alloc is ture when ptr is not nullptr."; 
            init_buffer(alloc, _data_type, need_alloc, ptr);
        }
    }        
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr)
                :_data_type(data_type), _dims(std::move(dims)) {
    _size = std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int32_t>());
    if(need_alloc && alloc) {   //需要分配并且有分配器
        allocate(alloc);
    } else {    //不需要分配，使用外部传入得内存
        if(ptr != nullptr) {    //外部传入ptr不空且need_alloc必须为false
            CHECK(need_alloc == false)
                << "The need_alloc is ture when ptr is not nullptr."; 
            init_buffer(alloc, _data_type, need_alloc, ptr);
        }
    }
}


bool Tensor::is_empty() const {
    return _buffer == nullptr || _size == 0 || _buffer->ptr() == nullptr;
}


int32_t Tensor::get_dim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->_dims.size());
    return this->_dims.at(idx);
}

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if(!_dims.empty()) {
        for(auto i = 0; i < _dims.size() - 1; i++) {
            size_t stride = std::accumulate(_dims.begin() + i + 1, _dims.end(), 1, std::multiplies<size_t>());
            strides.emplace_back(stride);
        }
        strides.emplace_back(1);
    }
    return strides;
}


void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type, 
                        bool need_alloc, void* ptr) {
    if(!alloc && !need_alloc) { //没有分配器 不需要alloc
        this->_buffer = std::make_shared<base::Buffer>(data_type_size(data_type) * _size, nullptr, ptr, true);
    } else {
        allocate(alloc, true);
    }
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {    //need_realloc表示重新分配
    if(!allocator) {
        LOG(ERROR) << "The allocator parameter in allocate function is null pointer!";
        return false;
    }

    size_t byte_size = this->byte_size();

    if(!byte_size) {
        LOG(ERROR) << "The byte size in allocate function is equall to zero!";
        return false;
    }

    if(_buffer && byte_size <= _buffer->byte_size()) {
        if(!need_realloc) {
            return true;
        }
    }
    _buffer = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    if(!_buffer->ptr()) {
        LOG(ERROR) << "The memory allocated is a null pointer!";
        return false;
    }
    return true;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(_dims.size()); }

}   //namespace tensor