#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);
    
    //根据申请显存大小处理
    if(byte_size > 1024 * 1024) {
        auto& big_buffers = _big_buffers_map[id];
        int sel_id = -1;
        for(int i = 0; i < big_buffers.size(); i++) {
            if(big_buffers[i]._byte_size >= byte_size && !big_buffers[i]._busy
            && big_buffers[i]._byte_size - byte_size < 1 * 1024 * 1024) {   
                //找空的且显存块大小与申请内存大小尽量接近的，减少内存的内部碎片
                if(sel_id == -1 || big_buffers[sel_id]._byte_size > big_buffers[i]._byte_size) {
                    sel_id = i;
                }
            }
        }
        
        if(sel_id != -1) {
            big_buffers[sel_id]._busy = true;
            return big_buffers[sel_id]._data;
        }

        //没找到空闲的 去cudamalloc
        void* ptr;
        state = cudaMalloc(&ptr, byte_size);
        if(state != cudaSuccess) {
            char buf[256];
            //snprintf 将格式化字符串写入buf
            snprintf(buf, 256, "Error: CUDA error when allocating %lu MB memory! maybe there's no"
                                "enough memoty left on device.", byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    //显存块大小小于1MB
    auto& cuda_buffers = _cuda_buffers_map[id];
    for(int i = 0; i < cuda_buffers.size(); i++) {
        if(cuda_buffers[i]._byte_size >= byte_size && !cuda_buffers[i]._busy) {
            cuda_buffers[i]._busy = true;
            _no_busy_cnt[id] -= cuda_buffers[i]._byte_size;
            return cuda_buffers[i]._data;
        }
    }
    
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
        char buf[256];
        snprintf(buf, 256,
                "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                "left on  device.",
                byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
}

void CUDADeviceAllocator::release(void *ptr) const {
    if(!ptr) {
        return; //防止二次析构
    }
    if(_cuda_buffers_map.empty()) {
        return;
    }
    
    //检查空闲显存块数量 若满足阈值则清理显存块 阈值设置为1GB
    //防止申请频率远大于释放频率时 系统剩余可以cudaMalloc的显存不足
    cudaError_t state = cudaSuccess;
    for(auto& it : _cuda_buffers_map) {
        if(_no_busy_cnt[it.first] > 1024 * 1024 * 1024) {
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for(int i = 0; i < cuda_buffers.size(); i++) {
                if(!cuda_buffers[i]._busy) {    //如果表项为空闲
                    state = cudaSetDevice(it.first);    //设置gpu到对应的显卡上
                    state = cudaFree(cuda_buffers[i]._data);
                    CHECK(state == cudaSuccess)
                        << "Error : CUDA error when release memory on device" << it.first;
                } else {
                    temp.push_back(cuda_buffers[i]);
                }
            }
            cuda_buffers.clear();
            it.second = temp;
            _no_busy_cnt[it.first] = 0;
        }
    }

    //释放ptr指向内存
    for(auto& it : _cuda_buffers_map) {
        auto& cuda_buffers = it.second;
        for(int i = 0; i < cuda_buffers.size(); i++) {
            if(cuda_buffers[i]._data == ptr) {
                cuda_buffers[i]._busy = false;
                _no_busy_cnt[it.first] += cuda_buffers[i]._byte_size;
                return;
            }
        }
        auto& big_buffers = _big_buffers_map[it.first];
        for(int i = 0; i < big_buffers.size(); i++) {
            if(big_buffers[i]._data == ptr) {
                big_buffers[i]._busy = false;
                return;
            }
        }
    }
    state = cudaFree(ptr);  //防御性编程 一般不会走到这一步
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

}   //namespace base