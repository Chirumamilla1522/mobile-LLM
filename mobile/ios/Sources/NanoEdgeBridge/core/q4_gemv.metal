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

// ============================================================================
// Kernel 1: High-Performance Q4_0 Matrix-Vector Multiply (GEMV)
// Optimized for LLM token decoding (single token input)
// Uses SIMDgroup hardware reductions and in-register dequantization
// ============================================================================
kernel void q4_0_gemv(
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
    // Each threadgroup computes 1 row of the output
    const uint row = tg_pos.x;
    if (row >= N) return;

    const uint blocks_per_row = K / 32;
    const device BlockQ4_0* row_weights = weights + (row * blocks_per_row);

    float thread_accum = 0.0f;

    // Threads in the threadgroup stride across the blocks of this row
    // Assuming 128 threads per threadgroup (4 SIMD groups of 32 threads)
    for (uint b = tid; b < blocks_per_row; b += 128) {
        const BlockQ4_0 block = row_weights[b];
        const float d = float(block.scale);
        const uint in_base = b * 32;

        // Unroll 16 bytes (32 4-bit weights)
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

    // Step 1: Fast hardware SIMD-group reduction across 32 lanes (zero shared memory latency)
    float simd_accum = simd_sum(thread_accum);

    // Step 2: Shared memory reduction across the 4 SIMD group leaders (128 threads = 4 simdgroups)
    threadgroup float simd_results[4];
    if (simd_lane == 0) {
        simd_results[simd_id] = simd_accum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Leader thread writes the final output
    if (tid == 0) {
        float final_sum = simd_results[0] + simd_results[1] + simd_results[2] + simd_results[3];
        output[row] = half(final_sum);
    }
}

// ============================================================================
// Kernel 2: MQ4_Apple Tiled Matrix-Vector Multiply
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

            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const uchar byte_val = tile.qs[qs_offset + j];
                const int q0 = int(byte_val & 0x0F) - 8;
                const int q1 = int(byte_val >> 4)   - 8;

                const float x0 = float(input[block_in_base + j * 2]);
                const float x1 = float(input[block_in_base + j * 2 + 1]);

                thread_accum += (float(q0) * scale) * x0;
                thread_accum += (float(q1) * scale) * x1;
            }
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
// Kernel 3: Fused RMSNorm
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
    // 128 threads compute variance of input vector
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
// Kernel 4: Vectorized Rotary Position Embedding (RoPE)
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
    const uint pair_idx = pos_in_grid.x; // 0 .. (head_dim/2 - 1)

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
