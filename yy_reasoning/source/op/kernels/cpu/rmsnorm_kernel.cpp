#include "rmsnorm_kernel.h"
 
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight, 
                        const tensor::Tensor& output, void* stream = nullptr){ 
    UNUSED(stream);
    //检查张量是否为空
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    //检察张量设备类型
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU && 
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

    const float* in_ptr = input.ptr<float>();
    const float* w_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
    arma::fvec w_tensor(const_cast<float*>(w_ptr), dim, false, true);

//根据编译时条件确定eps
#if defined(QWEM2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else 
    const float eps = 1e-5f;
#endif

    //使用指针直接对底层数据修改
    const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_tensor = w_tensor % (rsqrt * in_tensor);

}

}   //namespace kernel