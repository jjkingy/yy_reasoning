#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if(!byte_size) {
        return nullptr;
    } 
    void* data = malloc(byte_size);
    return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
    if(ptr) {
        free(ptr);
    }
}

}   //namespace base