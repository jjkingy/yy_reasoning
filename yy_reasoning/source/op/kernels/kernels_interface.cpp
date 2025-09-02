#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"
#include "cpu/emb_kernel.h"
#include "cuda/emb_kernel.cuh"
#include "cpu/matmul_kernel.h"
#include "cuda/matmul_kernel.cuh"
#include "cpu/swiglu_kernel.h"
#include "cuda/swiglu_kernel.cuh"
//mha_kernel cpu 未实现
// #include "cpu/mha_kernel.h"
#include "cuda/mha_kernel.cuh"
#include "kernels_interface.h"

namespace kernel {

AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return matmul_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return matmul_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rope_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return rope_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    if(device_type == base::DeviceType::kDeviceCUDA) {
        return matmul_kernel_cu_qint8;
    } else {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return mha_kernel;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return mha_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an mha kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type) {
    if(device_type == base::DeviceType::kDeviceCUDA) {
        return rmsnorm_kernel_cu_dim;
    } else {
        LOG(FATAL) << "Unknown device type for rmsnorm dim kernel";
        return nullptr;
    }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return emb_kernel_normal;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return emb_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an embedding kernel.";
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return swiglu_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return swiglu_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

}   //namespace kernel