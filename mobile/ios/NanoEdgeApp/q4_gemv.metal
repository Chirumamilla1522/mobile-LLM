#include <metal_stdlib>
using namespace metal;

// 18-byte Q4_0 Block definition matching host C++
struct BlockQ4_0 {
    half scale;
    uchar qs[16]; // 32 4-bit nibbles
};

// 144-byte MQ4_Apple Tile definition (8 blocks = 256 weights, cache-line aligned)
struct TileMQ4_Apple {
    half scales[8];
    uchar qs[128]; // 256 4-bit nibbles
};

// 34-byte Q8_0 Block definition matching host C++
struct BlockQ8_0 {
    half scale;
    char qs[32]; // 32 8-bit signed integers
};


// ============================================================================
// Kernel 1: Scale-Factored Q4_0 Matrix-Vector Multiply (GEMV)
// Accumulates integer/half dot products before applying block scale 'd'
// Eliminates 31 floating-point multiplications per 32-weight block
// ============================================================================
kernel void q4_0_gemv(
    device const BlockQ4_0*   weights     [[buffer(0)]], // [N, K/32]
    device const half*        input       [[buffer(1)]], // [K]
    device half*              output      [[buffer(2)]], // [N]
    constant uint&            K           [[buffer(3)]],
    constant uint&            N           [[buffer(4)]],
    uint2                     tg_pos      [[threadgroup_position_in_grid]],
    uint                      tid         [[thread_index_in_threadgroup]],
    uint                      simd_lane   [[thread_index_in_simdgroup]],
    uint                      simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= N) return;

    const uint blocks_per_row = K / 32;
    const device BlockQ4_0* row_weights = weights + (row * blocks_per_row);

    float thread_accum = 0.0f;

    for (uint b = tid; b < blocks_per_row; b += 128) {
        const BlockQ4_0 block = row_weights[b];
        const float d = float(block.scale);
        const uint in_base = b * 32;

        float block_dot = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            const uchar byte_val = block.qs[j];
            const int q0 = int(byte_val & 0x0F) - 8;
            const int q1 = int(byte_val >> 4)   - 8;

            const float x0 = float(input[in_base + j * 2]);
            const float x1 = float(input[in_base + j * 2 + 1]);

            block_dot += float(q0) * x0 + float(q1) * x1;
        }

        // Multiply scale ONCE per block
        thread_accum += block_dot * d;
    }

    float simd_accum = simd_sum(thread_accum);

    threadgroup float simd_results[4];
    if (simd_lane == 0) {
        simd_results[simd_id] = simd_accum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum = simd_results[0] + simd_results[1] + simd_results[2] + simd_results[3];
        output[row] = half(final_sum);
    }
}

// ============================================================================
// Kernel 1B: Multi-Row Tiled Q4_0 Matrix-Vector Multiply (2 Rows per Threadgroup)
// Reuses activation vector 'input' across 2 rows, cutting DRAM traffic by ~50%
// ============================================================================
kernel void q4_0_gemv_tiled2(
    device const BlockQ4_0*   weights     [[buffer(0)]], // [N, K/32]
    device const half*        input       [[buffer(1)]], // [K]
    device half*              output      [[buffer(2)]], // [N]
    constant uint&            K           [[buffer(3)]],
    constant uint&            N           [[buffer(4)]],
    uint2                     tg_pos      [[threadgroup_position_in_grid]],
    uint                      tid         [[thread_index_in_threadgroup]],
    uint                      simd_lane   [[thread_index_in_simdgroup]],
    uint                      simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = tg_pos.x * 2;
    const uint row1 = row0 + 1;
    if (row0 >= N) return;

    const uint blocks_per_row = K / 32;
    const device BlockQ4_0* row0_weights = weights + (row0 * blocks_per_row);
    const device BlockQ4_0* row1_weights = (row1 < N) ? weights + (row1 * blocks_per_row) : nullptr;

    float thread_accum0 = 0.0f;
    float thread_accum1 = 0.0f;

    for (uint b = tid; b < blocks_per_row; b += 128) {
        const BlockQ4_0 b0 = row0_weights[b];
        const float d0 = float(b0.scale);
        const uint in_base = b * 32;

        float block_dot0 = 0.0f;
        float block_dot1 = 0.0f;

        if (row1_weights != nullptr) {
            const BlockQ4_0 b1 = row1_weights[b];
            const float d1 = float(b1.scale);

            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                // Read input activation ONCE and reuse across both rows
                const float x0 = float(input[in_base + j * 2]);
                const float x1 = float(input[in_base + j * 2 + 1]);

                const uchar byte_val0 = b0.qs[j];
                const int q0_0 = int(byte_val0 & 0x0F) - 8;
                const int q1_0 = int(byte_val0 >> 4)   - 8;
                block_dot0 += float(q0_0) * x0 + float(q1_0) * x1;

                const uchar byte_val1 = b1.qs[j];
                const int q0_1 = int(byte_val1 & 0x0F) - 8;
                const int q1_1 = int(byte_val1 >> 4)   - 8;
                block_dot1 += float(q0_1) * x0 + float(q1_1) * x1;
            }

            thread_accum0 += block_dot0 * d0;
            thread_accum1 += block_dot1 * d1;
        } else {
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const float x0 = float(input[in_base + j * 2]);
                const float x1 = float(input[in_base + j * 2 + 1]);

                const uchar byte_val0 = b0.qs[j];
                const int q0_0 = int(byte_val0 & 0x0F) - 8;
                const int q1_0 = int(byte_val0 >> 4)   - 8;
                block_dot0 += float(q0_0) * x0 + float(q1_0) * x1;
            }
            thread_accum0 += block_dot0 * d0;
        }
    }

    float simd_accum0 = simd_sum(thread_accum0);
    float simd_accum1 = simd_sum(thread_accum1);

    threadgroup float simd_results0[4];
    threadgroup float simd_results1[4];
    if (simd_lane == 0) {
        simd_results0[simd_id] = simd_accum0;
        simd_results1[simd_id] = simd_accum1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum0 = simd_results0[0] + simd_results0[1] + simd_results0[2] + simd_results0[3];
        output[row0] = half(final_sum0);

        if (row1 < N) {
            float final_sum1 = simd_results1[0] + simd_results1[1] + simd_results1[2] + simd_results1[3];
            output[row1] = half(final_sum1);
        }
    }
}

// ============================================================================
// Kernel 1C: Baseline Q4_0 Matrix-Vector Multiply (Unoptimized for reference)
// ============================================================================
kernel void q4_0_gemv_baseline(
    device const BlockQ4_0*  weights     [[buffer(0)]], // [N, K/32]
    device const half*        input       [[buffer(1)]], // [K]
    device half*              output      [[buffer(2)]], // [N]
    constant uint&            K           [[buffer(3)]],
    constant uint&            N           [[buffer(4)]],
    uint2                     tg_pos      [[threadgroup_position_in_grid]],
    uint                      tid         [[thread_index_in_threadgroup]],
    uint                      simd_lane   [[thread_index_in_simdgroup]],
    uint                      simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= N) return;

    const uint blocks_per_row = K / 32;
    const device BlockQ4_0* row_weights = weights + (row * blocks_per_row);

    float thread_accum = 0.0f;

    for (uint b = tid; b < blocks_per_row; b += 128) {
        const BlockQ4_0 block = row_weights[b];
        const float d = float(block.scale);
        const uint in_base = b * 32;

        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            const uchar byte_val = block.qs[j];
            const int q0 = int(byte_val & 0x0F) - 8;
            const int q1 = int(byte_val >> 4)   - 8;

            const float x0 = float(input[in_base + j * 2]);
            const float x1 = float(input[in_base + j * 2 + 1]);

            thread_accum += (float(q0) * d) * x0;
            thread_accum += (float(q1) * d) * x1;
        }
    }

    float simd_accum = simd_sum(thread_accum);

    threadgroup float simd_results[4];
    if (simd_lane == 0) {
        simd_results[simd_id] = simd_accum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum = simd_results[0] + simd_results[1] + simd_results[2] + simd_results[3];
        output[row] = half(final_sum);
    }
}

// ============================================================================
// Kernel 2: MQ4_Apple Tiled Matrix-Vector Multiply (Scale-Factored)
// Reads 128-byte cache lines aligned to Apple Silicon unified memory
// ============================================================================
kernel void mq4_apple_gemv(
    device const TileMQ4_Apple* tiles       [[buffer(0)]], // [N, K/256]
    device const half*          input       [[buffer(1)]], // [K]
    device half*                output      [[buffer(2)]], // [N]
    constant uint&              K           [[buffer(3)]],
    constant uint&              N           [[buffer(4)]],
    uint2                       tg_pos      [[threadgroup_position_in_grid]],
    uint                        tid         [[thread_index_in_threadgroup]],
    uint                        simd_lane   [[thread_index_in_simdgroup]],
    uint                        simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= N) return;

    const uint tiles_per_row = K / 256;
    const device TileMQ4_Apple* row_tiles = tiles + (row * tiles_per_row);

    float thread_accum = 0.0f;

    for (uint t = tid; t < tiles_per_row; t += 128) {
        const TileMQ4_Apple tile = row_tiles[t];
        const uint tile_in_base = t * 256;

        #pragma unroll
        for (uint b = 0; b < 8; ++b) {
            const float scale = float(tile.scales[b]);
            const uint block_in_base = tile_in_base + b * 32;
            const uint qs_offset = b * 16;
            float block_dot = 0.0f;

            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const uchar byte_val = tile.qs[qs_offset + j];
                const int q0 = int(byte_val & 0x0F) - 8;
                const int q1 = int(byte_val >> 4)   - 8;

                const float x0 = float(input[block_in_base + j * 2]);
                const float x1 = float(input[block_in_base + j * 2 + 1]);

                block_dot += float(q0) * x0 + float(q1) * x1;
            }
            thread_accum += block_dot * scale;
        }
    }

    float simd_accum = simd_sum(thread_accum);

    threadgroup float simd_results[4];
    if (simd_lane == 0) {
        simd_results[simd_id] = simd_accum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum = simd_results[0] + simd_results[1] + simd_results[2] + simd_results[3];
        output[row] = half(final_sum);
    }
}

// ============================================================================
// Kernel 2B: MQ4_Apple Tiled Matrix-Vector Multiply (2 Rows per Threadgroup)
// ============================================================================
kernel void mq4_apple_gemv_tiled2(
    device const TileMQ4_Apple* tiles       [[buffer(0)]], // [N, K/256]
    device const half*          input       [[buffer(1)]], // [K]
    device half*                output      [[buffer(2)]], // [N]
    constant uint&              K           [[buffer(3)]],
    constant uint&              N           [[buffer(4)]],
    uint2                       tg_pos      [[threadgroup_position_in_grid]],
    uint                        tid         [[thread_index_in_threadgroup]],
    uint                        simd_lane   [[thread_index_in_simdgroup]],
    uint                        simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = tg_pos.x * 2;
    const uint row1 = row0 + 1;
    if (row0 >= N) return;

    const uint tiles_per_row = K / 256;
    const device TileMQ4_Apple* row0_tiles = tiles + (row0 * tiles_per_row);
    const device TileMQ4_Apple* row1_tiles = (row1 < N) ? tiles + (row1 * tiles_per_row) : nullptr;

    float thread_accum0 = 0.0f;
    float thread_accum1 = 0.0f;

    for (uint t = tid; t < tiles_per_row; t += 128) {
        const TileMQ4_Apple tile0 = row0_tiles[t];
        const uint tile_in_base = t * 256;

        if (row1_tiles != nullptr) {
            const TileMQ4_Apple tile1 = row1_tiles[t];

            #pragma unroll
            for (uint b = 0; b < 8; ++b) {
                const float scale0 = float(tile0.scales[b]);
                const float scale1 = float(tile1.scales[b]);
                const uint block_in_base = tile_in_base + b * 32;
                const uint qs_offset = b * 16;

                float block_dot0 = 0.0f;
                float block_dot1 = 0.0f;

                #pragma unroll
                for (uint j = 0; j < 16; ++j) {
                    const float x0 = float(input[block_in_base + j * 2]);
                    const float x1 = float(input[block_in_base + j * 2 + 1]);

                    const uchar byte_val0 = tile0.qs[qs_offset + j];
                    const int q0_0 = int(byte_val0 & 0x0F) - 8;
                    const int q1_0 = int(byte_val0 >> 4)   - 8;
                    block_dot0 += float(q0_0) * x0 + float(q1_0) * x1;

                    const uchar byte_val1 = tile1.qs[qs_offset + j];
                    const int q0_1 = int(byte_val1 & 0x0F) - 8;
                    const int q1_1 = int(byte_val1 >> 4)   - 8;
                    block_dot1 += float(q0_1) * x0 + float(q1_1) * x1;
                }

                thread_accum0 += block_dot0 * scale0;
                thread_accum1 += block_dot1 * scale1;
            }
        } else {
            #pragma unroll
            for (uint b = 0; b < 8; ++b) {
                const float scale0 = float(tile0.scales[b]);
                const uint block_in_base = tile_in_base + b * 32;
                const uint qs_offset = b * 16;

                float block_dot0 = 0.0f;

                #pragma unroll
                for (uint j = 0; j < 16; ++j) {
                    const float x0 = float(input[block_in_base + j * 2]);
                    const float x1 = float(input[block_in_base + j * 2 + 1]);

                    const uchar byte_val0 = tile0.qs[qs_offset + j];
                    const int q0_0 = int(byte_val0 & 0x0F) - 8;
                    const int q1_0 = int(byte_val0 >> 4)   - 8;
                    block_dot0 += float(q0_0) * x0 + float(q1_0) * x1;
                }
                thread_accum0 += block_dot0 * scale0;
            }
        }
    }

    float simd_accum0 = simd_sum(thread_accum0);
    float simd_accum1 = simd_sum(thread_accum1);

    threadgroup float simd_results0[4];
    threadgroup float simd_results1[4];
    if (simd_lane == 0) {
        simd_results0[simd_id] = simd_accum0;
        simd_results1[simd_id] = simd_accum1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum0 = simd_results0[0] + simd_results0[1] + simd_results0[2] + simd_results0[3];
        output[row0] = half(final_sum0);

        if (row1 < N) {
            float final_sum1 = simd_results1[0] + simd_results1[1] + simd_results1[2] + simd_results1[3];
            output[row1] = half(final_sum1);
        }
    }
}

// ============================================================================
// Kernel 3: High-Performance Q8_0 (INT8) Matrix-Vector Multiply (GEMV)
// ============================================================================
kernel void q8_0_gemv(
    device const BlockQ8_0*   weights     [[buffer(0)]], // [N, K/32]
    device const half*        input       [[buffer(1)]], // [K]
    device half*              output      [[buffer(2)]], // [N]
    constant uint&            K           [[buffer(3)]],
    constant uint&            N           [[buffer(4)]],
    uint2                     tg_pos      [[threadgroup_position_in_grid]],
    uint                      tid         [[thread_index_in_threadgroup]],
    uint                      simd_lane   [[thread_index_in_simdgroup]],
    uint                      simd_id     [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= N) return;

    const uint blocks_per_row = K / 32;
    const device BlockQ8_0* row_weights = weights + (row * blocks_per_row);

    float thread_accum = 0.0f;

    for (uint b = tid; b < blocks_per_row; b += 128) {
        const BlockQ8_0 block = row_weights[b];
        const float d = float(block.scale);
        const uint in_base = b * 32;

        float block_dot = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 32; ++j) {
            const float q = float(block.qs[j]);
            const float x = float(input[in_base + j]);
            block_dot += q * x;
        }

        thread_accum += block_dot * d;
    }

    float simd_accum = simd_sum(thread_accum);

    threadgroup float simd_results[4];
    if (simd_lane == 0) {
        simd_results[simd_id] = simd_accum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum = simd_results[0] + simd_results[1] + simd_results[2] + simd_results[3];
        output[row] = half(final_sum);
    }
}

// ============================================================================
// Kernel 4: Fused RMSNorm
// y = (x / sqrt(mean(x^2) + eps)) * weight
// ============================================================================
kernel void rms_norm(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    constant uint&     dim     [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    uint               tid     [[thread_index_in_threadgroup]],
    uint               simd_lane [[thread_index_in_simdgroup]],
    uint               simd_id   [[simdgroup_index_in_threadgroup]]
) {
    float thread_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = float(input[i]);
        thread_sum_sq += v * v;
    }

    float simd_sq = simd_sum(thread_sum_sq);

    threadgroup float shared_sq[4];
    if (simd_lane == 0) {
        shared_sq[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float inv_rms;
    if (tid == 0) {
        float mean_sq = (shared_sq[0] + shared_sq[1] + shared_sq[2] + shared_sq[3]) / float(dim);
        inv_rms = rsqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = inv_rms;
    for (uint i = tid; i < dim; i += 128) {
        output[i] = half(float(input[i]) * scale * float(weight[i]));
    }
}

// ============================================================================
// Kernel 5: Vectorized Rotary Position Embedding (RoPE)
// ============================================================================
kernel void rope_embedding(
    device half*    q_or_k      [[buffer(0)]],
    constant uint&  head_dim    [[buffer(1)]],
    constant uint&  num_heads   [[buffer(2)]],
    constant uint&  pos         [[buffer(3)]],
    constant float& theta_base  [[buffer(4)]],
    uint2           pos_in_grid [[thread_position_in_grid]]
) {
    const uint head_idx = pos_in_grid.y;
    const uint pair_idx = pos_in_grid.x;

    if (head_idx >= num_heads || pair_idx >= head_dim / 2) return;

    const float freq = 1.0f / pow(theta_base, float(pair_idx * 2) / float(head_dim));
    const float angle = float(pos) * freq;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    const uint base = head_idx * head_dim + pair_idx * 2;
    const float x0 = float(q_or_k[base]);
    const float x1 = float(q_or_k[base + 1]);

    q_or_k[base]     = half(x0 * cos_val - x1 * sin_val);
    q_or_k[base + 1] = half(x0 * sin_val + x1 * cos_val);
}
