#pragma once

#include "mllm/types.h"
#include <cstdint>

namespace mllm {

class NeonGemv {
public:
    // Multi-threaded GCD accelerated CPU NEON GEMV
    static void compute_q4_0(
        const BlockQ4_0* weights,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    // Single-core reference implementation for baseline comparison
    static void compute_q4_0_single_core(
        const BlockQ4_0* weights,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    static void compute_mq4_apple(
        const TileMQ4_Apple* tiles,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    static void compute_mq4_apple_single_core(
        const TileMQ4_Apple* tiles,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    static void compute_q8_0(
        const BlockQ8_0* weights,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    static void compute_q8_0_single_core(
        const BlockQ8_0* weights,
        const float* input,
        float* output,
        uint32_t K,
        uint32_t N
    );

    static void rms_norm(
        const float* input,
        const float* weight,
        float* output,
        uint32_t dim,
        float eps = 1e-5f
    );
};

} // namespace mllm
