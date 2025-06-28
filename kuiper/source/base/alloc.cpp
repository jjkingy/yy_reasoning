#include"base/alloc.h"
#include<cuda_runtime_api.h>
#include"logging.h"
#include<cstring>

namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                            MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if(!byte_size) {
        return;
    }
    cudaStream_t _stream = nullptr;
    if(stream) {
        _stream = static_cast<cudaStream_t>(stream);
    }
    if(memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    }else if(memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        if(!_stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        }else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, _stream);
        }
    }else if(memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        if(!_stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        }else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, _stream);
        }
    }else if(memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        if(!_stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        }else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, _stream);
        }
    }else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    if(need_sync) {
        cudaDeviceSynchronize();
    }
}


void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
    CHECK(_device_type != DeviceType::kDeviceUnknown);
    if(_device_type == DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    }else {
        if(stream) {
            cudaStream_t _stream = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, _stream);
        }else {
            cudaMemset(ptr, 0, byte_size);
        }
        if(need_sync) {
            cudaDeviceSynchronize();
        }
    }
}

} //namespace base