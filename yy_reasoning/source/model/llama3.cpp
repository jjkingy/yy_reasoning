#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>

namespace model {

void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if(add_layer_) {
    _add_layer_->set_cuda_config(config);
    _add_layer_->to_cuda();
    }

    if(_rope_layer) {
        _rope_layer->set_cuda_config(config);
        _rope_layer->to_cuda();
    }

    if(_swiglu_layer) {
        _swiglu_layer->set_cuda_config(config);
        _swiglu_layer->to_cuda();
    }

    if(_cls_layer) {
        _cls_layer->set_cuda_config(config);
        _cls_layer->to_cuda();
    }

    if(_embedding_layer) {
        _embedding_layer->set_cuda_config(config);
        _embedding_layer->to_cuda();
    }

    if(_mha_layer) {
        _mha_layer->set_cuda_config(config);
        _mha_layer->to_cuda();
    }

    for (auto& weight_layer : _wq_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wk_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wv_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _wo_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w1_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w2_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : _w3_layers) {
        if(weight_layer) {
        weight_layer->set_cuda_config(config);
        weight_layer->to_cuda();
        }
    }

    for (auto& rms_norm_layer : _rmsnorm_layers) {
        if(rms_norm_layer) {
        rms_norm_layer->to_cuda();
        rms_norm_layer->set_cuda_config(config);
        }
    }
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

base::Status create_layers() {
    if()
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