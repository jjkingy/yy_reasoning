#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if(!byte_size) {
        return nullptr;
    }
    void* data = nullptr;
    cudaError_t err = cudaMalloc(&data, data);
    CHECK_EQ(err, cudaSuccess);
    return data;
}

void CUDADeviceAllocator::release(*ptr) const {
    if(ptr) {
        cudaFree(ptr);
    }
}

}