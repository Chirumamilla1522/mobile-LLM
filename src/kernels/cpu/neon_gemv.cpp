#include "neon_gemv.hpp"
#include <cmath>
#include <algorithm>
#include <thread>

#if defined(__ARM_NEON) || defined(__arm64__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace mllm {

static uint32_t gemv_worker_count(uint32_t rows) {
    const uint32_t cores = std::max(1u, std::thread::hardware_concurrency());
    return std::max(1u, std::min({4u, cores, (rows + 255u) / 256u}));
}

// Single-core Q4_0 reference implementation
void NeonGemv::compute_q4_0_single_core(
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

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ4_0& block = row_blocks[b];
            const float scale = fp16_to_fp32(block.scale);
            const float* in_ptr = input + (b * 32);

            float b_sum = 0.0f;
            for (uint32_t j = 0; j < 16; ++j) {
                uint8_t byte_val = block.qs[j];
                int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                int q1 = static_cast<int>(byte_val >> 4) - 8;

                b_sum += static_cast<float>(q0) * in_ptr[j * 2] + static_cast<float>(q1) * in_ptr[j * 2 + 1];
            }
            row_sum += b_sum * scale;
        }
        output[r] = row_sum;
    }
}

// Multi-threaded GCD accelerated Q4_0 NEON GEMV
void NeonGemv::compute_q4_0(
    const BlockQ4_0* weights,
    const float* input,
    float* output,
    uint32_t K,
    uint32_t N
) {
    const uint32_t blocks_per_row = K / 32;

#if defined(__APPLE__)
    const uint32_t num_threads = gemv_worker_count(N);
    const uint32_t rows_per_thread = (N + num_threads - 1) / num_threads;

    dispatch_apply(num_threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t_idx) {
        const uint32_t r_start = static_cast<uint32_t>(t_idx * rows_per_thread);
        const uint32_t r_end = std::min(r_start + rows_per_thread, N);

        for (uint32_t r = r_start; r < r_end; ++r) {
            const BlockQ4_0* row_blocks = weights + (r * blocks_per_row);
            float row_sum = 0.0f;

#if defined(__ARM_NEON)
            float32x4_t row_accum_v = vdupq_n_f32(0.0f);

            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const BlockQ4_0& block = row_blocks[b];
                const float scale = fp16_to_fp32(block.scale);
                const float* in_ptr = input + (b * 32);

                // Load 16 bytes (32 4-bit weights)
                const uint8x16_t packed = vld1q_u8(block.qs);
                const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
                const uint8x16_t low_u8  = vandq_u8(packed, mask_0f);
                const uint8x16_t high_u8 = vshrq_n_u8(packed, 4);

                const int8x16_t offset_8 = vdupq_n_s8(8);
                const int8x16_t q_even = vsubq_s8(vreinterpretq_s8_u8(low_u8), offset_8);
                const int8x16_t q_odd  = vsubq_s8(vreinterpretq_s8_u8(high_u8), offset_8);

                int8x16x2_t zipped = vzipq_s8(q_even, q_odd);

                // In-register SIMD dot product: zero stack spilling
                float32x4_t block_accum = vdupq_n_f32(0.0f);

                int16x8_t w0_lo = vmovl_s8(vget_low_s8(zipped.val[0]));
                int16x8_t w0_hi = vmovl_s8(vget_high_s8(zipped.val[0]));

                float32x4_t w_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_lo)));
                float32x4_t w_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_lo)));
                float32x4_t w_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_hi)));
                float32x4_t w_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_hi)));

                block_accum = vmlaq_f32(block_accum, w_f0, vld1q_f32(in_ptr + 0));
                block_accum = vmlaq_f32(block_accum, w_f1, vld1q_f32(in_ptr + 4));
                block_accum = vmlaq_f32(block_accum, w_f2, vld1q_f32(in_ptr + 8));
                block_accum = vmlaq_f32(block_accum, w_f3, vld1q_f32(in_ptr + 12));

                int16x8_t w1_lo = vmovl_s8(vget_low_s8(zipped.val[1]));
                int16x8_t w1_hi = vmovl_s8(vget_high_s8(zipped.val[1]));

                float32x4_t w_f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_lo)));
                float32x4_t w_f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_lo)));
                float32x4_t w_f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_hi)));
                float32x4_t w_f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_hi)));

                block_accum = vmlaq_f32(block_accum, w_f4, vld1q_f32(in_ptr + 16));
                block_accum = vmlaq_f32(block_accum, w_f5, vld1q_f32(in_ptr + 20));
                block_accum = vmlaq_f32(block_accum, w_f6, vld1q_f32(in_ptr + 24));
                block_accum = vmlaq_f32(block_accum, w_f7, vld1q_f32(in_ptr + 28));

                // Scale factored out per block
                row_accum_v = vmlaq_n_f32(row_accum_v, block_accum, scale);
            }
            row_sum = vaddvq_f32(row_accum_v);
#else
            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const BlockQ4_0& block = row_blocks[b];
                const float scale = fp16_to_fp32(block.scale);
                const float* in_ptr = input + (b * 32);

                float b_sum = 0.0f;
                for (uint32_t j = 0; j < 16; ++j) {
                    uint8_t byte_val = block.qs[j];
                    int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                    int q1 = static_cast<int>(byte_val >> 4) - 8;
                    b_sum += static_cast<float>(q0) * in_ptr[j * 2] + static_cast<float>(q1) * in_ptr[j * 2 + 1];
                }
                row_sum += b_sum * scale;
            }
#endif
            output[r] = row_sum;
        }
    });
#else
    compute_q4_0_single_core(weights, input, output, K, N);
#endif
}

void NeonGemv::compute_mq4_apple_single_core(
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

                float b_sum = 0.0f;
                for (uint32_t j = 0; j < 16; ++j) {
                    uint8_t byte_val = tile.qs[qs_offset + j];
                    int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                    int q1 = static_cast<int>(byte_val >> 4) - 8;

                    b_sum += static_cast<float>(q0) * block_in[j * 2] + static_cast<float>(q1) * block_in[j * 2 + 1];
                }
                row_sum += b_sum * scale;
            }
        }
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

#if defined(__APPLE__)
    const uint32_t num_threads = gemv_worker_count(N);
    const uint32_t rows_per_thread = (N + num_threads - 1) / num_threads;

    dispatch_apply(num_threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t_idx) {
        const uint32_t r_start = static_cast<uint32_t>(t_idx * rows_per_thread);
        const uint32_t r_end = std::min(r_start + rows_per_thread, N);

        for (uint32_t r = r_start; r < r_end; ++r) {
            const TileMQ4_Apple* row_tiles = tiles + (r * tiles_per_row);
            float row_sum = 0.0f;

            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                const TileMQ4_Apple& tile = row_tiles[t];
                const float* tile_in = input + (t * 256);

                for (uint32_t b = 0; b < 8; ++b) {
                    const float scale = fp16_to_fp32(tile.scales[b]);
                    const float* block_in = tile_in + (b * 32);
                    const uint32_t qs_offset = b * 16;

                    float b_sum = 0.0f;
                    for (uint32_t j = 0; j < 16; ++j) {
                        uint8_t byte_val = tile.qs[qs_offset + j];
                        int q0 = static_cast<int>(byte_val & 0x0F) - 8;
                        int q1 = static_cast<int>(byte_val >> 4) - 8;

                        b_sum += static_cast<float>(q0) * block_in[j * 2] + static_cast<float>(q1) * block_in[j * 2 + 1];
                    }
                    row_sum += b_sum * scale;
                }
            }
            output[r] = row_sum;
        }
    });
#else
    compute_mq4_apple_single_core(tiles, input, output, K, N);
#endif
}

void NeonGemv::compute_q8_0_single_core(
    const BlockQ8_0* weights,
    const float* input,
    float* output,
    uint32_t K,
    uint32_t N
) {
    const uint32_t blocks_per_row = K / 32;

    for (uint32_t r = 0; r < N; ++r) {
        const BlockQ8_0* row_blocks = weights + (r * blocks_per_row);
        float row_sum = 0.0f;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ8_0& block = row_blocks[b];
            float scale = fp16_to_fp32(block.scale);
            const float* in_ptr = input + (b * 32);

            float block_sum = 0.0f;
            for (uint32_t j = 0; j < 32; ++j) {
                block_sum += static_cast<float>(block.qs[j]) * in_ptr[j];
            }
            row_sum += block_sum * scale;
        }
        output[r] = row_sum;
    }
}

void NeonGemv::compute_q8_0(
    const BlockQ8_0* weights,
    const float* input,
    float* output,
    uint32_t K,
    uint32_t N
) {
    const uint32_t blocks_per_row = K / 32;

#if defined(__APPLE__)
    const uint32_t num_threads = gemv_worker_count(N);
    const uint32_t rows_per_thread = (N + num_threads - 1) / num_threads;

    dispatch_apply(num_threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t_idx) {
        const uint32_t r_start = static_cast<uint32_t>(t_idx * rows_per_thread);
        const uint32_t r_end = std::min(r_start + rows_per_thread, N);

        for (uint32_t r = r_start; r < r_end; ++r) {
            const BlockQ8_0* row_blocks = weights + (r * blocks_per_row);
            float row_sum = 0.0f;

            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const BlockQ8_0& block = row_blocks[b];
                float scale = fp16_to_fp32(block.scale);
                const float* in_ptr = input + (b * 32);

                float block_sum = 0.0f;
                for (uint32_t j = 0; j < 32; ++j) {
                    block_sum += static_cast<float>(block.qs[j]) * in_ptr[j];
                }
                row_sum += block_sum * scale;
            }
            output[r] = row_sum;
        }
    });
#else
    compute_q8_0_single_core(weights, input, output, K, N);
#endif
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
