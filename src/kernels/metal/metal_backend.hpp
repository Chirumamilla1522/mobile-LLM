#pragma once

#include "mllm/types.h"
#include <string>
#include <memory>
#include <chrono>

namespace mllm {

struct DeviceInfo {
    std::string device_name;
    bool has_unified_memory{false};
    uint64_t max_buffer_length{0};
    uint32_t max_threads_per_threadgroup{0};
};

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    // Device metadata
    [[nodiscard]] const DeviceInfo& device_info() const noexcept;

    // Zero-copy buffer creation from mmap pointers
    // NOTE: bytes must be page-aligned (e.g. 16KB on Apple Silicon) and length a multiple of page size
    void* create_buffer_no_copy(const void* bytes, size_t length);
    void release_buffer(void* buffer_handle);

    // Allocate managed/shared GPU memory
    void* allocate_buffer(size_t length);
    void* get_buffer_contents(void* buffer_handle);

    // Kernel execution routines
    // Returns execution time in microseconds
    double dispatch_q4_0_gemv(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_q4_0_gemv_tiled2(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_q4_0_gemv_baseline(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_mq4_apple_gemv(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_mq4_apple_gemv_tiled2(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_q8_0_gemv(
        void* weights_buffer,
        void* input_buffer,
        void* output_buffer,
        uint32_t K,
        uint32_t N
    );

    double dispatch_rms_norm(
        void* input_buffer,
        void* weight_buffer,
        void* output_buffer,
        uint32_t dim,
        float eps = 1e-5f
    );

    double dispatch_rope(
        void* q_or_k_buffer,
        uint32_t head_dim,
        uint32_t num_heads,
        uint32_t pos,
        float theta_base = 10000.0f
    );

    // Synchronize command queue
    void synchronize();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace mllm
