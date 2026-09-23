#include "neon_gemv.hpp"
#include <cmath>

#if defined(__ARM_NEON) || defined(__arm64__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm {

void NeonGemv::compute_q4_0(
    const BlockQ4_0* weights,
    const float* input,
    float* output,
    uint32_t K,
    uint32_t N
) {
    const uint32_t blocks_per_row = K / 32;

    for (uint32_t r = 0; r < N; ++r) {
        const BlockQ4_0* row_blocks = weights + (r * blocks_per_row);
        float row_sum = 0.0f;

#if defined(__ARM_NEON)
        float32x4_t accum_v0 = vdupq_n_f32(0.0f);
        float32x4_t accum_v1 = vdupq_n_f32(0.0f);

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_0& block = row_blocks[b];
            const float scale = fp16_to_fp32(block.scale);
            const float* in_ptr = input + (b * 32);

            // Load 16 bytes = 32 4-bit nibbles
            const uint8x16_t packed = vld1q_u8(block.qs);

            // Extract low and high nibbles
            const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
            const uint8x16_t low_u8  = vandq_u8(packed, mask_0f);
            const uint8x16_t high_u8 = vshrq_n_u8(packed, 4);

            // Subtract 8 to center signed range [-8, 7]
            const int8x16_t offset_8 = vdupq_n_s8(8);
            const int8x16_t q_even_s8 = vsubq_s8(vreinterpretq_s8_u8(low_u8), offset_8);
            const int8x16_t q_odd_s8  = vsubq_s8(vreinterpretq_s8_u8(high_u8), offset_8);

            // Interleave even and odd back into sequential 32 weights
            int8x16x2_t zip_lo = vzipq_s8(q_even_s8, q_odd_s8); // weights 0..15 and 16..31

            // Multiply each block by input activations
            // Convert to 16-bit integers, then float, then scale and accumulate
            int8_t w_buf[32];
            vst1q_s8(w_buf, zip_lo.val[0]);
            vst1q_s8(w_buf + 16, zip_lo.val[1]);

            for (int i = 0; i < 32; i += 4) {
                float32x4_t in_v = vld1q_f32(in_ptr + i);
                float32x4_t w_v = {
                    static_cast<float>(w_buf[i]) * scale,
                    static_cast<float>(w_buf[i + 1]) * scale,
                    static_cast<float>(w_buf[i + 2]) * scale,
                    static_cast<float>(w_buf[i + 3]) * scale
                };
                accum_v0 = vmlaq_f32(accum_v0, in_v, w_v);
            }
        }
        accum_v0 = vaddq_f32(accum_v0, accum_v1);
        row_sum = vaddvq_f32(accum_v0);
#else
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_0& block = row_blocks[b];
            const float scale = fp16_to_fp32(block.scale);
            const float* in_ptr = input + (b * 32);

            for (uint32_t j = 0; j < 16; ++j) {
                uint8_t byte_val = block.qs[j];
                int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                int q1 = static_cast<int>(byte_val >> 4) - 8;

                row_sum += (static_cast<float>(q0) * scale) * in_ptr[j * 2];
                row_sum += (static_cast<float>(q1) * scale) * in_ptr[j * 2 + 1];
            }
        }
#endif
        output[r] = row_sum;
    }
}

void NeonGemv::compute_mq4_apple(
    const TileMQ4_Apple* tiles,
    const float* input,
    float* output,
    uint32_t K,
    uint32_t N
) {
    const uint32_t tiles_per_row = K / 256;

    for (uint32_t r = 0; r < N; ++r) {
        const TileMQ4_Apple* row_tiles = tiles + (r * tiles_per_row);
        float row_sum = 0.0f;

        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            const TileMQ4_Apple& tile = row_tiles[t];
            const float* tile_in = input + (t * 256);

            for (uint32_t b = 0; b < 8; ++b) {
                const float scale = fp16_to_fp32(tile.scales[b]);
                const float* block_in = tile_in + (b * 32);
                const uint32_t qs_offset = b * 16;

                for (uint32_t j = 0; j < 16; ++j) {
                    uint8_t byte_val = tile.qs[qs_offset + j];
                    int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                    int q1 = static_cast<int>(byte_val >> 4) - 8;

                    row_sum += (static_cast<float>(q0) * scale) * block_in[j * 2];
                    row_sum += (static_cast<float>(q1) * scale) * block_in[j * 2 + 1];
                }
            }
        }
        output[r] = row_sum;
    }
}

void NeonGemv::rms_norm(
    const float* input,
    const float* weight,
    float* output,
    uint32_t dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        sum_sq += input[i] * input[i];
    }
    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);

    for (uint32_t i = 0; i < dim; ++i) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

} // namespace mllm
