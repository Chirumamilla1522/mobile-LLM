#pragma once

#include "mllm/types.h"
#include <cstdint>

namespace mllm {

class NeonGemv {
public:
    // CPU reference and vectorized NEON implementation of Q4_0 Matrix-Vector multiplication
    // W: [N, K/32] in BlockQ4_0
    // x: [K] in FP16 or FP32
    // y: [N] in FP32
    static void compute_q4_0(
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

    static void rms_norm(
        const float* input,
        const float* weight,
        float* output,
        uint32_t dim,
        float eps = 1e-5f
    );
};

} // namespace mllm
