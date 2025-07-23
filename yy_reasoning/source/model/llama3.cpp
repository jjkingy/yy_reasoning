#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>

namespace model {

    //未完成
void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if(_add_layer) {
        _add_layer->set_cuda_config(config);
        _add_layer->to_cuda();
    }

    if(_rope_layer) {
        _rope_layer->set_cuda_config(config);
        _rope_layer->to_cuda();
    }

    ////////
}

//未完成
//参数检查 → 设备初始化 → 读权重 → 分配内存 → 预计算 RoPE → 生成采样器
base::Status LLama2Model::init(base::DeviceType device_type) {
    // using namespace base;
    //参数检查 token和设备合法性
    if(_token_path.empty()) {
        return base::error::PathNotValid(_token_path);
    }
    if(device == base::DeviceType::kDeviceCPU && _is_quant_model) {
        return base::error::InternalError("The cpu don't support quant model.");
    }

    //设备初始化
    _device_type = device_type;
    if(device_type == base::DeviceType::kDeviceCUDA) {
        cuadSetDevcie(0);   //设置GPU
        _cuda_config = std::make_shared<kernel::CudaConfig>();  //把GPU句柄放在cudaconfig一起管理
        cudaStreamCreate(&_cuda_config->stream);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            return error::InternalError("The cuda hanle create failed.");
        }
    }

    //读权重
    base::Status read_status = gen_model_from_file();
    if(!read_status) {
        return read_status;
    }

    //分配内存
    init_mem();

    //预计算ROPE
    if(_device_type == base::DeviceType::kDeviceCPU) {

    }

    //生成采样器

    return error::Success();
}

//未完成
void LLama2Model::init_mem() {

    //设置内存分配器
    if(_device_type == base::DeviceType::kDeviceCPU) {
        auto alloc = base::AllocatorFactory<base::CPUDeviceAllocator>::get_instance();
    } else {
        auto alloc = base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();
    }

    //如果是cuda就转移到gpu上
    if(_device_type = base::DeviceType::kDeviceCUDA) {
        CHECK_NE(_cuda_config, nullptr);
        _llama_layers->to_cuda(_cuda_config);
    }

    auto alloc_cpu = base::AllocatorFactory<base::CPUDeviceAllocator>::get_instance();
    auto alloc_cu = base::AllocatorFactory<base::CUDADeviceAllocator>::get_instance();

    //未完成， 分配各层Tensor并注册insert到buffer

    

}

}   //namespace model