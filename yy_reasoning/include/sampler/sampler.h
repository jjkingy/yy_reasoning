#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>

namespace sampler {
class Sampler {
public:
    explicit Sampler(base::DeviceType device_type) : _device_type(device_type) {}

    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;
protected:
    base::DeviceType _device_type;

};

}   //sampler
#endif