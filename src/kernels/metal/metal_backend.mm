#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.hpp"
#include "mllm/model_format.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstring>

namespace mllm {

// Embedded fallback MSL shader source code
static const char* EMBEDDED_METAL_SOURCE = R"metal(
#include <metal_stdlib>
using namespace metal;

struct BlockQ4_0 {
    half scale;
    uchar qs[16];
};

struct TileMQ4_Apple {
    half scales[8];
    uchar qs[128];
};

struct BlockQ8_0 {
    half scale;
    char qs[32];
};

struct BlockKVQ8 {
    half scale;
    char values[32];
};

// 1. Scale-Factored Q4_0 Matrix-Vector Multiply (GEMV)
kernel void q4_0_gemv(
    device const BlockQ4_0*   weights     [[buffer(0)]],
    device const half*        input       [[buffer(1)]],
    device half*              output      [[buffer(2)]],
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

// 1B. 2-Row Tiled Q4_0 Matrix-Vector Multiply
kernel void q4_0_gemv_tiled2(
    device const BlockQ4_0*   weights     [[buffer(0)]],
    device const half*        input       [[buffer(1)]],
    device half*              output      [[buffer(2)]],
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

// 1C. Baseline Unoptimized Q4_0 Matrix-Vector Multiply
kernel void q4_0_gemv_baseline(
    device const BlockQ4_0*  weights     [[buffer(0)]],
    device const half*        input       [[buffer(1)]],
    device half*              output      [[buffer(2)]],
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

// 2. Scale-Factored MQ4_Apple Tiled Matrix-Vector Multiply
kernel void mq4_apple_gemv(
    device const TileMQ4_Apple* tiles       [[buffer(0)]],
    device const half*          input       [[buffer(1)]],
    device half*                output      [[buffer(2)]],
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

// 2B. 2-Row Tiled MQ4_Apple Matrix-Vector Multiply
kernel void mq4_apple_gemv_tiled2(
    device const TileMQ4_Apple* tiles       [[buffer(0)]],
    device const half*          input       [[buffer(1)]],
    device half*                output      [[buffer(2)]],
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

// 3. High-Performance Q8_0 (INT8) Matrix-Vector Multiply (GEMV)
kernel void q8_0_gemv(
    device const BlockQ8_0*   weights     [[buffer(0)]],
    device const half*        input       [[buffer(1)]],
    device half*              output      [[buffer(2)]],
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

// 4. Fused RMSNorm
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

    if (tid == 0) {
        float mean_sq = (shared_sq[0] + shared_sq[1] + shared_sq[2] + shared_sq[3]) / float(dim);
        shared_sq[0] = rsqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = shared_sq[0];
    for (uint i = tid; i < dim; i += 128) {
        output[i] = half(float(input[i]) * scale * float(weight[i]));
    }
}

// 5. Vectorized Rotary Position Embedding (RoPE)
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

    const uint base = head_idx * head_dim + pair_idx;
    const float x0 = float(q_or_k[base]);
    const float x1 = float(q_or_k[base + head_dim / 2]);

    q_or_k[base]     = half(x0 * cos_val - x1 * sin_val);
    q_or_k[base + head_dim / 2] = half(x0 * sin_val + x1 * cos_val);
}

kernel void embedding_fp16(
    device const half* weights [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < dim) output[i] = weights[i];
}

kernel void embedding_q4(
    device const BlockQ4_0* weights [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= dim) return;
    const BlockQ4_0 block = weights[i / 32];
    const uchar packed = block.qs[(i % 32) / 2];
    const int q = (i & 1) ? int(packed >> 4) - 8 : int(packed & 15) - 8;
    output[i] = half(float(q) * float(block.scale));
}

kernel void embedding_mq4(
    device const TileMQ4_Apple* weights [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= dim) return;
    const TileMQ4_Apple tile = weights[i / 256];
    const uint within = i % 256;
    const uint block = within / 32;
    const uchar packed = tile.qs[block * 16 + (within % 32) / 2];
    const int q = (within & 1) ? int(packed >> 4) - 8 : int(packed & 15) - 8;
    output[i] = half(float(q) * float(tile.scales[block]));
}

kernel void add_inplace(
    device half* x [[buffer(0)]],
    device const half* update [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < dim) x[i] = half(float(x[i]) + float(update[i]));
}

kernel void rope_embedding_table(
    device half* q_or_k [[buffer(0)]],
    device const half* cos_table [[buffer(1)]],
    device const half* sin_table [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= num_heads || gid.x >= head_dim / 2) return;
    const uint table_index = pos * (head_dim / 2) + gid.x;
    const float c = float(cos_table[table_index]);
    const float s = float(sin_table[table_index]);
    const uint base = gid.y * head_dim + gid.x;
    const float x0 = float(q_or_k[base]);
    const float x1 = float(q_or_k[base + head_dim / 2]);
    q_or_k[base] = half(x0 * c - x1 * s);
    q_or_k[base + head_dim / 2] = half(x0 * s + x1 * c);
}

kernel void store_kv(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device half* cache_k [[buffer(2)]],
    device half* cache_v [[buffer(3)]],
    constant uint& layer [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    const uint kv_dim = kv_heads * head_dim;
    if (i >= kv_dim) return;
    const uint h = i / head_dim;
    const uint d = i % head_dim;
    const uint dst = ((layer * kv_heads + h) * context + position) * head_dim + d;
    cache_k[dst] = k[i];
    cache_v[dst] = v[i];
}

kernel void decode_attention_online(
    device const half* q [[buffer(0)]],
    device const half* cache_k [[buffer(1)]],
    device const half* cache_v [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& layer [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (head >= query_heads) return;
    const uint kv_head = head / (query_heads / kv_heads);
    const uint cache_base = (layer * kv_heads + kv_head) * context * head_dim;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    const float scale = rsqrt(float(head_dim));
    for (uint t = 0; t <= position; ++t) {
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32)
            partial += float(q[head * head_dim + d]) * float(cache_k[cache_base + t * head_dim + d]);
        const float score = simd_sum(partial) * scale;
        const float next_max = max(maximum, score);
        const float old_scale = exp(maximum - next_max);
        const float value_scale = exp(score - next_max);
        for (uint j = 0; j < 4; ++j) {
            const uint d = lane + j * 32;
            if (d < head_dim)
                accum[j] = accum[j] * old_scale + float(cache_v[cache_base + t * head_dim + d]) * value_scale;
        }
        normalizer = normalizer * old_scale + value_scale;
        maximum = next_max;
    }
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + j * 32;
        if (d < head_dim) output[head * head_dim + d] = half(accum[j] / normalizer);
    }
}

kernel void store_kv_q8(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device BlockKVQ8* cache_k [[buffer(2)]],
    device BlockKVQ8* cache_v [[buffer(3)]],
    constant uint& layer [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint blocks_per_head = (head_dim + 31) / 32;
    const uint block = group.x;
    const uint head = group.y;
    if (block >= blocks_per_head || head >= kv_heads) return;
    const uint d = block * 32 + lane;
    const float kval = d < head_dim ? float(k[head * head_dim + d]) : 0.0f;
    const float vval = d < head_dim ? float(v[head * head_dim + d]) : 0.0f;
    const float kscale = max(simd_max(abs(kval)) / 127.0f, 1e-8f);
    const float vscale = max(simd_max(abs(vval)) / 127.0f, 1e-8f);
    const uint index = ((layer * kv_heads + head) * context + position) * blocks_per_head + block;
    if (lane == 0) { cache_k[index].scale = half(kscale); cache_v[index].scale = half(vscale); }
    cache_k[index].values[lane] = char(clamp(rint(kval / kscale), -127.0f, 127.0f));
    cache_v[index].values[lane] = char(clamp(rint(vval / vscale), -127.0f, 127.0f));
}

kernel void decode_attention_online_q8(
    device const half* q [[buffer(0)]],
    device const BlockKVQ8* cache_k [[buffer(1)]],
    device const BlockKVQ8* cache_v [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& layer [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (head >= query_heads) return;
    const uint kv_head = head / (query_heads / kv_heads);
    const uint blocks_per_head = (head_dim + 31) / 32;
    const uint cache_base = (layer * kv_heads + kv_head) * context * blocks_per_head;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    const float attention_scale = rsqrt(float(head_dim));
    for (uint t = 0; t <= position; ++t) {
        float partial = 0.0f;
        for (uint block = 0; block < blocks_per_head; ++block) {
            const uint d = block * 32 + lane;
            const BlockKVQ8 kb = cache_k[cache_base + t * blocks_per_head + block];
            if (d < head_dim) partial += float(q[head * head_dim + d]) * float(kb.values[lane]) * float(kb.scale);
        }
        const float score = simd_sum(partial) * attention_scale;
        const float next_max = max(maximum, score);
        const float old_scale = exp(maximum - next_max);
        const float value_scale = exp(score - next_max);
        for (uint block = 0; block < min(blocks_per_head, 4u); ++block) {
            const uint d = block * 32 + lane;
            const BlockKVQ8 vb = cache_v[cache_base + t * blocks_per_head + block];
            if (d < head_dim) accum[block] = accum[block] * old_scale + float(vb.values[lane]) * float(vb.scale) * value_scale;
        }
        normalizer = normalizer * old_scale + value_scale;
        maximum = next_max;
    }
    for (uint block = 0; block < min(blocks_per_head, 4u); ++block) {
        const uint d = block * 32 + lane;
        if (d < head_dim) output[head * head_dim + d] = half(accum[block] / normalizer);
    }
}

kernel void q4_0_swiglu(
    device const BlockQ4_0* gate [[buffer(0)]],
    device const BlockQ4_0* up [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    if (row >= N) return;
    const uint blocks = K / 32;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < blocks; b += 128) {
        const BlockQ4_0 gb = gate[row * blocks + b];
        const BlockQ4_0 ub = up[row * blocks + b];
        float gd = 0.0f, ud = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            const float x0 = float(input[b * 32 + j * 2]);
            const float x1 = float(input[b * 32 + j * 2 + 1]);
            gd += (float(int(gb.qs[j] & 15) - 8) * x0 + float(int(gb.qs[j] >> 4) - 8) * x1);
            ud += (float(int(ub.qs[j] & 15) - 8) * x0 + float(int(ub.qs[j] >> 4) - 8) * x1);
        }
        gate_sum += gd * float(gb.scale);
        up_sum += ud * float(ub.scale);
    }
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    threadgroup float gs[4], us[4];
    if (lane == 0) { gs[simd_id] = gate_sum; us[simd_id] = up_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float g = gs[0] + gs[1] + gs[2] + gs[3];
        const float u = us[0] + us[1] + us[2] + us[3];
        output[row] = half((g / (1.0f + exp(-g))) * u);
    }
}

kernel void swiglu_elementwise(
    device const half* gate [[buffer(0)]],
    device half* up_and_output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < dim) {
        const float g = float(gate[i]);
        up_and_output[i] = half((g / (1.0f + exp(-g))) * float(up_and_output[i]));
    }
}

inline float sampling_logit(device const half* logits, device const int* recent,
                            uint recent_count, uint index, float temperature,
                            float repetition_penalty) {
    float value = float(logits[index]);
    for (uint i = 0; i < recent_count; ++i) {
        if (recent[i] == int(index)) {
            value = value > 0.0f ? value / repetition_penalty : value * repetition_penalty;
            break;
        }
    }
    return value / max(temperature, 0.01f);
}

kernel void sample_min_p(
    device const half* logits [[buffer(0)]],
    device const int* recent [[buffer(1)]],
    device atomic_uint* selected [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& recent_count [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    constant float& repetition_penalty [[buffer(6)]],
    constant float& min_p [[buffer(7)]],
    constant float& random_value [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[257];
    threadgroup uint indices[256];
    threadgroup float global_max;
    const uint chunk = (vocab + 255) / 256;
    const uint begin = min(tid * chunk, vocab);
    const uint end = min(begin + chunk, vocab);
    float local_max = -INFINITY;
    uint local_index = 0;
    for (uint i = begin; i < end; ++i) {
        const float value = sampling_logit(logits, recent, recent_count, i, temperature, repetition_penalty);
        if (value > local_max) { local_max = value; local_index = i; }
    }
    values[tid] = local_max; indices[tid] = local_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = values[0]; uint best_index = indices[0];
        for (uint i = 1; i < 256; ++i) if (values[i] > best) { best = values[i]; best_index = indices[i]; }
        global_max = best;
        atomic_store_explicit(selected, temperature <= 0.05f ? best_index : vocab, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (temperature <= 0.05f) return;
    float local_sum = 0.0f;
    for (uint i = begin; i < end; ++i) {
        const float probability = exp(sampling_logit(logits, recent, recent_count, i, temperature, repetition_penalty) - global_max);
        if (probability >= min_p) local_sum += probability;
    }
    values[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 256; ++i) { const float value = values[i]; values[i] = sum; sum += value; }
        values[256] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float target = random_value * values[256];
    if (target >= values[tid] && target < values[tid] + local_sum) {
        float running = values[tid];
        for (uint i = begin; i < end; ++i) {
            const float probability = exp(sampling_logit(logits, recent, recent_count, i, temperature, repetition_penalty) - global_max);
            if (probability >= min_p) running += probability;
            if (running >= target) { atomic_fetch_min_explicit(selected, i, memory_order_relaxed); break; }
        }
    }
}

kernel void embedding_batch(
    device const half* weights [[buffer(0)]], device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < dim) output[gid.y * dim + gid.x] = weights[tokens[gid.y] * dim + gid.x];
}

kernel void embedding_q4_batch(
    device const BlockQ4_0* weights [[buffer(0)]], device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& dim [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=dim)return; const uint i=tokens[gid.y]*dim+gid.x; const BlockQ4_0 b=weights[i/32];
    const uchar p=b.qs[(i%32)/2]; const int q=(i&1)?int(p>>4)-8:int(p&15)-8;
    output[gid.y*dim+gid.x]=half(float(q)*float(b.scale));
}

kernel void embedding_mq4_batch(
    device const TileMQ4_Apple* weights [[buffer(0)]], device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& dim [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=dim)return; const uint i=tokens[gid.y]*dim+gid.x, within=i%256; const TileMQ4_Apple tile=weights[i/256];
    const uint block=within/32; const uchar p=tile.qs[block*16+(within%32)/2]; const int q=(within&1)?int(p>>4)-8:int(p&15)-8;
    output[gid.y*dim+gid.x]=half(float(q)*float(tile.scales[block]));
}

kernel void rms_norm_batch(
    device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& dim [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += 128) { const float x = float(input[token * dim + i]); sum += x * x; }
    sum = simd_sum(sum);
    threadgroup float partial[4], inverse;
    if (lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) inverse = rsqrt((partial[0] + partial[1] + partial[2] + partial[3]) / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < dim; i += 128) output[token * dim + i] = half(float(input[token * dim + i]) * inverse * float(weight[i]));
}

kernel void q4_0_gemm_t4x2(
    device const BlockQ4_0* weights [[buffer(0)]], device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& batch [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group.x * 2, row1 = row0 + 1, token0 = group.y * 4;
    if (row0 >= N || token0 >= batch) return;
    const uint blocks = K / 32;
    float sums[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    for (uint b = tid; b < blocks; b += 128) {
        const BlockQ4_0 w0 = weights[row0 * blocks + b];
        const bool has_row1 = row1 < N;
        const BlockQ4_0 w1 = has_row1 ? weights[row1 * blocks + b] : w0;
        for (uint j = 0; j < 16; ++j) {
            const int a0 = int(w0.qs[j] & 15) - 8, a1 = int(w0.qs[j] >> 4) - 8;
            const int b0 = int(w1.qs[j] & 15) - 8, b1 = int(w1.qs[j] >> 4) - 8;
            for (uint t = 0; t < 4; ++t) if (token0 + t < batch) {
                const uint base = (token0 + t) * K + b * 32 + j * 2;
                const float x0 = float(input[base]), x1 = float(input[base + 1]);
                sums[t] += (float(a0) * x0 + float(a1) * x1) * float(w0.scale);
                if (has_row1) sums[4 + t] += (float(b0) * x0 + float(b1) * x1) * float(w1.scale);
            }
        }
    }
    threadgroup float reductions[32];
    for (uint i = 0; i < 8; ++i) {
        const float value = simd_sum(sums[i]);
        if (lane == 0) reductions[i * 4 + simd_id] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) for (uint i = 0; i < 8; ++i) {
        const uint token = token0 + (i & 3), row = i < 4 ? row0 : row1;
        if (token < batch && row < N)
            output[token * N + row] = half(reductions[i*4] + reductions[i*4+1] + reductions[i*4+2] + reductions[i*4+3]);
    }
}

kernel void rope_batch(
    device half* values [[buffer(0)]], device const half* cos_table [[buffer(1)]], device const half* sin_table [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]], constant uint& heads [[buffer(4)]],
    constant uint& start_pos [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= head_dim / 2 || gid.y >= heads) return;
    const uint token = gid.z, pos = start_pos + token, half_dim = head_dim / 2;
    const uint base = token * heads * head_dim + gid.y * head_dim + gid.x;
    const float x0 = float(values[base]), x1 = float(values[base + half_dim]);
    const float c = float(cos_table[pos * half_dim + gid.x]), s = float(sin_table[pos * half_dim + gid.x]);
    values[base] = half(x0*c - x1*s); values[base + half_dim] = half(x0*s + x1*c);
}

kernel void store_kv_q8_batch(
    device const half* k [[buffer(0)]], device const half* v [[buffer(1)]],
    device BlockKVQ8* cache_k [[buffer(2)]], device BlockKVQ8* cache_v [[buffer(3)]],
    constant uint& layer [[buffer(4)]], constant uint& start_pos [[buffer(5)]], constant uint& context [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]], constant uint& head_dim [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint blocks = (head_dim + 31) / 32, block = group.x, head = group.y, token = group.z;
    const uint d = block * 32 + lane, source = token * kv_heads * head_dim + head * head_dim + d;
    const float kval = d < head_dim ? float(k[source]) : 0.0f, vval = d < head_dim ? float(v[source]) : 0.0f;
    const float ks = max(simd_max(abs(kval))/127.0f, 1e-8f), vs = max(simd_max(abs(vval))/127.0f, 1e-8f);
    const uint dst = ((layer * kv_heads + head) * context + start_pos + token) * blocks + block;
    if (lane == 0) { cache_k[dst].scale = half(ks); cache_v[dst].scale = half(vs); }
    cache_k[dst].values[lane] = char(clamp(rint(kval/ks),-127.0f,127.0f));
    cache_v[dst].values[lane] = char(clamp(rint(vval/vs),-127.0f,127.0f));
}

kernel void attention_q8_batch(
    device const half* q [[buffer(0)]], device const BlockKVQ8* cache_k [[buffer(1)]],
    device const BlockKVQ8* cache_v [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& layer [[buffer(4)]], constant uint& start_pos [[buffer(5)]], constant uint& context [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]], constant uint& kv_heads [[buffer(8)]], constant uint& head_dim [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x, token = group.y, position = start_pos + token;
    if (head >= query_heads) return;
    const uint kv_head = head/(query_heads/kv_heads), blocks=(head_dim+31)/32;
    const uint cache_base=(layer*kv_heads+kv_head)*context*blocks;
    float accum[4]={0,0,0,0}, maximum=-INFINITY, normalizer=0.0f;
    for(uint t=0;t<=position;++t){
        float partial=0.0f;
        for(uint b=0;b<blocks;++b){const uint d=b*32+lane; const BlockKVQ8 kb=cache_k[cache_base+t*blocks+b];
            if(d<head_dim) partial+=float(q[token*query_heads*head_dim+head*head_dim+d])*float(kb.values[lane])*float(kb.scale);}
        const float score=simd_sum(partial)*rsqrt(float(head_dim)), next=max(maximum,score);
        const float old=exp(maximum-next), fresh=exp(score-next);
        for(uint b=0;b<min(blocks,4u);++b){const uint d=b*32+lane; const BlockKVQ8 vb=cache_v[cache_base+t*blocks+b];
            if(d<head_dim) accum[b]=accum[b]*old+float(vb.values[lane])*float(vb.scale)*fresh;}
        normalizer=normalizer*old+fresh; maximum=next;
    }
    for(uint b=0;b<min(blocks,4u);++b){const uint d=b*32+lane; if(d<head_dim) output[token*query_heads*head_dim+head*head_dim+d]=half(accum[b]/normalizer);}
}

kernel void add_batch(device half* x [[buffer(0)]], device const half* update [[buffer(1)]],
                      constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<count) x[i]=half(float(x[i])+float(update[i]));
}

kernel void swiglu_batch(device const half* gate [[buffer(0)]], device half* up [[buffer(1)]],
                         constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<count){const float g=float(gate[i]); up[i]=half((g/(1.0f+exp(-g)))*float(up[i]));}
}
)metal";

struct MetalBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLLibrary> library{nil};

    id<MTLComputePipelineState> pso_q4_0_gemv{nil};
    id<MTLComputePipelineState> pso_q4_0_gemv_tiled2{nil};
    id<MTLComputePipelineState> pso_q4_0_gemv_baseline{nil};
    id<MTLComputePipelineState> pso_mq4_apple_gemv{nil};
    id<MTLComputePipelineState> pso_mq4_apple_gemv_tiled2{nil};
    id<MTLComputePipelineState> pso_q8_0_gemv{nil};
    id<MTLComputePipelineState> pso_rms_norm{nil};
    id<MTLComputePipelineState> pso_rope{nil};

    DeviceInfo device_info{};
};

MetalBackend::MetalBackend() : pimpl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        pimpl_->device = MTLCreateSystemDefaultDevice();
        if (!pimpl_->device) {
            throw std::runtime_error("No Metal-capable GPU found.");
        }

        pimpl_->command_queue = [pimpl_->device newCommandQueue];
        if (!pimpl_->command_queue) {
            throw std::runtime_error("Failed to create Metal command queue.");
        }

        pimpl_->device_info.device_name = std::string([[pimpl_->device name] UTF8String]);
        pimpl_->device_info.has_unified_memory = [pimpl_->device hasUnifiedMemory];
        pimpl_->device_info.max_buffer_length = [pimpl_->device maxBufferLength];
        pimpl_->device_info.max_threads_per_threadgroup = 1024;

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:EMBEDDED_METAL_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop

        pimpl_->library = [pimpl_->device newDefaultLibrary];
        if (!pimpl_->library)
            pimpl_->library = [pimpl_->device newLibraryWithSource:source options:options error:&error];
        if (!pimpl_->library) {
            std::string err_msg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            throw std::runtime_error("Metal shader compilation failed: " + err_msg);
        }

        auto create_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [pimpl_->library newFunctionWithName:name];
            if (!fn) {
                throw std::runtime_error("Metal function not found: " + std::string([name UTF8String]));
            }
            NSError* pso_err = nil;
            id<MTLComputePipelineState> pso = [pimpl_->device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!pso) {
                std::string err_msg = pso_err ? [[pso_err localizedDescription] UTF8String] : "unknown error";
                throw std::runtime_error("Failed to create PSO for " + std::string([name UTF8String]) + ": " + err_msg);
            }
            return pso;
        };

        pimpl_->pso_q4_0_gemv = create_pso(@"q4_0_gemv");
        pimpl_->pso_q4_0_gemv_tiled2 = create_pso(@"q4_0_gemv_tiled2");
        pimpl_->pso_q4_0_gemv_baseline = create_pso(@"q4_0_gemv_baseline");
        pimpl_->pso_mq4_apple_gemv = create_pso(@"mq4_apple_gemv");
        pimpl_->pso_mq4_apple_gemv_tiled2 = create_pso(@"mq4_apple_gemv_tiled2");
        pimpl_->pso_q8_0_gemv = create_pso(@"q8_0_gemv");
        pimpl_->pso_rms_norm = create_pso(@"rms_norm");
        pimpl_->pso_rope = create_pso(@"rope_embedding");
    }
}

MetalBackend::~MetalBackend() {
}

const DeviceInfo& MetalBackend::device_info() const noexcept {
    return pimpl_->device_info;
}

void* MetalBackend::create_buffer_no_copy(const void* bytes, size_t length) {
    @autoreleasepool {
        size_t page_aligned_len = (length + 16383) & ~16383;
        id<MTLBuffer> buf = [pimpl_->device newBufferWithBytesNoCopy:(void*)bytes
                                                              length:page_aligned_len
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        if (!buf) {
            std::cerr << "Warning: newBufferWithBytesNoCopy failed, falling back to allocated copy" << std::endl;
            buf = [pimpl_->device newBufferWithBytes:bytes
                                              length:length
                                             options:MTLResourceStorageModeShared];
        }
        return (__bridge_retained void*)buf;
    }
}

void* MetalBackend::allocate_buffer(size_t length) {
    @autoreleasepool {
        size_t page_aligned_len = (length + 16383) & ~16383;
        if (page_aligned_len == 0) page_aligned_len = 16384;
        id<MTLBuffer> buf = [pimpl_->device newBufferWithLength:page_aligned_len
                                                        options:MTLResourceStorageModeShared];
        return (__bridge_retained void*)buf;
    }
}

void* MetalBackend::get_buffer_contents(void* buffer_handle) {
    if (!buffer_handle) return nullptr;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_handle;
    return [buf contents];
}

void MetalBackend::release_buffer(void* buffer_handle) {
    if (buffer_handle) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer_handle;
        (void)buf;
    }
}

double MetalBackend::dispatch_q4_0_gemv(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_q4_0_gemv];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(N, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_q4_0_gemv_tiled2(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_q4_0_gemv_tiled2];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake((N + 1) / 2, 1, 1); // 2 rows per threadgroup

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_q4_0_gemv_baseline(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_q4_0_gemv_baseline];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(N, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_mq4_apple_gemv(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_mq4_apple_gemv];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(N, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_mq4_apple_gemv_tiled2(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_mq4_apple_gemv_tiled2];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake((N + 1) / 2, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_q8_0_gemv(
    void* weights_buffer,
    void* input_buffer,
    void* output_buffer,
    uint32_t K,
    uint32_t N
) {
    @autoreleasepool {
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weights_buffer;
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_q8_0_gemv];
        [encoder setBuffer:w_buf offset:0 atIndex:0];
        [encoder setBuffer:in_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(N, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_rms_norm(
    void* input_buffer,
    void* weight_buffer,
    void* output_buffer,
    uint32_t dim,
    float eps
) {
    @autoreleasepool {
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weight_buffer;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_rms_norm];
        [encoder setBuffer:in_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&eps length:sizeof(float) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(1, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

double MetalBackend::dispatch_rope(
    void* q_or_k_buffer,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t pos,
    float theta_base
) {
    @autoreleasepool {
        id<MTLBuffer> qk_buf = (__bridge id<MTLBuffer>)q_or_k_buffer;

        id<MTLCommandBuffer> cmd_buf = [pimpl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pimpl_->pso_rope];
        [encoder setBuffer:qk_buf offset:0 atIndex:0];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:1];
        [encoder setBytes:&num_heads length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&pos length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&theta_base length:sizeof(float) atIndex:4];

        MTLSize threads_per_tg = MTLSizeMake(32, 1, 1);
        MTLSize grid_size = MTLSizeMake((head_dim / 2 + 31) / 32, num_heads, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_tg];
        [encoder endEncoding];

        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::micro>(end - start).count();
    }
}

void MetalBackend::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [pimpl_->command_queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace mllm

struct NanoEdgeMetalLayer {
    const mllm::TensorDescriptor *in_norm{}, *q{}, *k{}, *v{}, *o{};
    const mllm::TensorDescriptor *post_norm{}, *gate{}, *up{}, *down{};
};

struct NanoEdgeMetalBridge {
    const mllm::ModelHeader* header{};
    const mllm::TensorDescriptor* descriptors{};
    std::vector<NanoEdgeMetalLayer> layers;
    const mllm::TensorDescriptor *embedding{}, *final_norm{}, *lm_head{};
    uint32_t context{}, head_dim{}, kv_dim{};
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> queue{nil};
    id<MTLLibrary> library{nil};
    id<MTLBuffer> model{nil}, x{nil}, norm{nil}, q{nil}, k{nil}, v{nil};
    id<MTLBuffer> attn{nil}, projection{nil}, gate{nil}, ffn{nil}, logits{nil};
    id<MTLBuffer> cache_k{nil}, cache_v{nil}, rope_cos{nil}, rope_sin{nil}, recent{nil}, sampled_token{nil};
    id<MTLBuffer> batch_x{nil}, batch_norm{nil}, batch_q{nil}, batch_k{nil}, batch_v{nil};
    id<MTLBuffer> batch_attn{nil}, batch_projection{nil}, batch_gate{nil}, batch_ffn{nil};
    id<MTLComputePipelineState> p_embedding_fp16{nil}, p_embedding_q4{nil}, p_embedding_mq4{nil};
    id<MTLComputePipelineState> p_q4{nil}, p_mq4{nil}, p_q8{nil}, p_norm{nil}, p_rope{nil};
    id<MTLComputePipelineState> p_store_kv_fp16{nil}, p_attention_fp16{nil}, p_store_kv_q8{nil}, p_attention_q8{nil};
    id<MTLComputePipelineState> p_add{nil}, p_swiglu{nil}, p_swiglu_element{nil}, p_sampler{nil};
    id<MTLComputePipelineState> p_embed_batch{nil}, p_embed_q4_batch{nil}, p_embed_mq4_batch{nil};
    id<MTLComputePipelineState> p_norm_batch{nil}, p_gemm_batch{nil}, p_rope_batch{nil};
    id<MTLComputePipelineState> p_store_batch{nil}, p_attention_batch{nil}, p_add_batch{nil}, p_swiglu_batch{nil};
    uint64_t rng_state{0x853c49e6748fea9bULL};
    uint32_t prefill_hidden_limit{2048}; // ponytail: M3 crossover; override after physical-device calibration.
    bool q8_kv{true};
    size_t kv_cache_size{};

    void configure_kv_cache(bool use_q8) {
        if (cache_k && q8_kv == use_q8) return;
        if (queue && cache_k) { id<MTLCommandBuffer> command=[queue commandBuffer]; [command commit]; [command waitUntilCompleted]; }
        q8_kv = use_q8;
        const size_t vectors=static_cast<size_t>(header->num_layers)*header->num_kv_heads*context;
        kv_cache_size = use_q8 ? vectors*((head_dim+31)/32)*34 : vectors*head_dim*sizeof(uint16_t);
        cache_k=[device newBufferWithLength:kv_cache_size options:MTLResourceStorageModeShared];
        cache_v=[device newBufferWithLength:kv_cache_size options:MTLResourceStorageModeShared];
        if(!cache_k||!cache_v) throw std::runtime_error("KV cache allocation failed");
    }

    NanoEdgeMetalBridge(const void* bytes, size_t length, uint32_t context_capacity, float rope_theta) {
        if (!bytes || length < sizeof(mllm::ModelHeader)) throw std::runtime_error("invalid model");
        header = static_cast<const mllm::ModelHeader*>(bytes);
        if (header->magic != mllm::MLLM_MAGIC || header->manifest_offset + header->manifest_size > length)
            throw std::runtime_error("invalid model manifest");
        descriptors = reinterpret_cast<const mllm::TensorDescriptor*>(static_cast<const uint8_t*>(bytes) + header->manifest_offset);
        context = std::min(context_capacity, header->max_seq_len);
        head_dim = header->hidden_dim / header->num_heads;
        kv_dim = header->num_kv_heads * head_dim;
        if (!context || !head_dim || head_dim > 128 || (head_dim & 1) || header->hidden_dim % 32 || header->intermediate_dim % 32)
            throw std::runtime_error("unsupported model dimensions");

        layers.resize(header->num_layers);
        for (uint32_t i = 0; i < header->num_tensors; ++i) {
            const auto* d = &descriptors[i];
            if (d->offset + d->size_bytes > length) throw std::runtime_error("tensor outside model");
            if (d->layer_idx >= 0 && static_cast<uint32_t>(d->layer_idx) < header->num_layers) {
                auto& l = layers[d->layer_idx];
                switch (static_cast<mllm::TensorType>(d->tensor_type)) {
                    case mllm::TensorType::ATTN_Q: l.q = d; break;
                    case mllm::TensorType::ATTN_K: l.k = d; break;
                    case mllm::TensorType::ATTN_V: l.v = d; break;
                    case mllm::TensorType::ATTN_OUT: l.o = d; break;
                    case mllm::TensorType::ATTN_NORM: l.in_norm = d; break;
                    case mllm::TensorType::FFN_GATE: l.gate = d; break;
                    case mllm::TensorType::FFN_UP: l.up = d; break;
                    case mllm::TensorType::FFN_DOWN: l.down = d; break;
                    case mllm::TensorType::FFN_NORM: l.post_norm = d; break;
                    default: break;
                }
            } else {
                switch (static_cast<mllm::TensorType>(d->tensor_type)) {
                    case mllm::TensorType::EMBEDDINGS: embedding = d; break;
                    case mllm::TensorType::FINAL_NORM: final_norm = d; break;
                    case mllm::TensorType::LM_HEAD: lm_head = d; break;
                    default: break;
                }
            }
        }
        if (!lm_head) lm_head = embedding;
        if (!embedding || !final_norm || !lm_head) throw std::runtime_error("missing global tensor");
        for (const auto& l : layers) {
            if (!l.in_norm || !l.q || !l.k || !l.v || !l.o || !l.post_norm || !l.gate || !l.up || !l.down)
                throw std::runtime_error("missing layer tensor");
            const auto supported = [](uint32_t type) { return type == 2 || type == 3 || type == 5; };
            if (!supported(l.q->quant_type) || !supported(l.k->quant_type) || !supported(l.v->quant_type) ||
                !supported(l.o->quant_type) || !supported(l.gate->quant_type) ||
                !supported(l.up->quant_type) || !supported(l.down->quant_type))
                throw std::runtime_error("unsupported projection format");
        }
        if ((embedding->quant_type != 1 && embedding->quant_type != 3 && embedding->quant_type != 5) ||
            (lm_head->quant_type != 2 && lm_head->quant_type != 3 && lm_head->quant_type != 5))
            throw std::runtime_error("unsupported embedding or LM head");

        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        NSError* error = nil;
        library = [device newDefaultLibrary];
        if (!library)
            library = [device newLibraryWithSource:[NSString stringWithUTF8String:mllm::EMBEDDED_METAL_SOURCE] options:nil error:&error];
        if (!library) throw std::runtime_error(error ? [[error localizedDescription] UTF8String] : "Metal compile failed");
        auto pipeline = [&](NSString* name) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
            if (!result) throw std::runtime_error([[error localizedDescription] UTF8String]);
            return result;
        };
        p_embedding_fp16 = pipeline(@"embedding_fp16"); p_embedding_q4 = pipeline(@"embedding_q4");
        p_embedding_mq4 = pipeline(@"embedding_mq4");
        p_q4 = pipeline(@"q4_0_gemv_tiled2"); p_mq4 = pipeline(@"mq4_apple_gemv_tiled2"); p_q8 = pipeline(@"q8_0_gemv");
        p_norm = pipeline(@"rms_norm"); p_rope = pipeline(@"rope_embedding_table");
        p_store_kv_fp16 = pipeline(@"store_kv"); p_attention_fp16 = pipeline(@"decode_attention_online");
        p_store_kv_q8 = pipeline(@"store_kv_q8"); p_attention_q8 = pipeline(@"decode_attention_online_q8");
        p_add = pipeline(@"add_inplace"); p_swiglu = pipeline(@"q4_0_swiglu");
        p_swiglu_element = pipeline(@"swiglu_elementwise");
        p_sampler = pipeline(@"sample_min_p");
        p_embed_batch = pipeline(@"embedding_batch"); p_embed_q4_batch = pipeline(@"embedding_q4_batch");
        p_embed_mq4_batch = pipeline(@"embedding_mq4_batch"); p_norm_batch = pipeline(@"rms_norm_batch");
        p_gemm_batch = pipeline(@"q4_0_gemm_t4x2"); p_rope_batch = pipeline(@"rope_batch");
        p_store_batch = pipeline(@"store_kv_q8_batch"); p_attention_batch = pipeline(@"attention_q8_batch");
        p_add_batch = pipeline(@"add_batch"); p_swiglu_batch = pipeline(@"swiglu_batch");

        model = [device newBufferWithBytesNoCopy:const_cast<void*>(bytes) length:length options:MTLResourceStorageModeShared deallocator:nil];
        if (!model) model = [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
        auto buffer = [&](size_t count) { return [device newBufferWithLength:count * sizeof(uint16_t) options:MTLResourceStorageModeShared]; };
        x = buffer(header->hidden_dim); norm = buffer(header->hidden_dim); q = buffer(header->hidden_dim);
        k = buffer(kv_dim); v = buffer(kv_dim); attn = buffer(header->hidden_dim);
        projection = buffer(header->hidden_dim); gate = buffer(header->intermediate_dim);
        ffn = buffer(header->intermediate_dim); logits = buffer(header->vocab_size);
        configure_kv_cache(true);
        const size_t rope_count = static_cast<size_t>(context) * head_dim / 2;
        rope_cos = buffer(rope_count); rope_sin = buffer(rope_count);
        recent = [device newBufferWithLength:64 * sizeof(int32_t) options:MTLResourceStorageModeShared];
        sampled_token = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        constexpr size_t batch_capacity = 32;
        batch_x = buffer(batch_capacity * header->hidden_dim); batch_norm = buffer(batch_capacity * header->hidden_dim);
        batch_q = buffer(batch_capacity * header->hidden_dim); batch_k = buffer(batch_capacity * kv_dim);
        batch_v = buffer(batch_capacity * kv_dim); batch_attn = buffer(batch_capacity * header->hidden_dim);
        batch_projection = buffer(batch_capacity * header->hidden_dim); batch_gate = buffer(batch_capacity * header->intermediate_dim);
        batch_ffn = buffer(batch_capacity * header->intermediate_dim);
        auto* c = static_cast<uint16_t*>([rope_cos contents]);
        auto* s = static_cast<uint16_t*>([rope_sin contents]);
        for (uint32_t pos = 0; pos < context; ++pos) for (uint32_t pair = 0; pair < head_dim / 2; ++pair) {
            const float angle = float(pos) / std::pow(rope_theta, float(pair * 2) / float(head_dim));
            c[pos * head_dim / 2 + pair] = mllm::fp32_to_fp16(std::cos(angle));
            s[pos * head_dim / 2 + pair] = mllm::fp32_to_fp16(std::sin(angle));
        }
    }

    bool forward(uint32_t token, uint32_t pos, bool compute_logits, float* cpu_logits,
                 const int32_t* sample_tokens = nullptr, uint32_t sample_count = 0,
                 float temperature = 0.0f, float penalty = 1.0f, float min_p = 0.0f,
                 int32_t* next_token = nullptr) {
        if (token >= header->vocab_size || pos >= context) return false;
        const bool sampling = next_token != nullptr;
        sample_count = std::min(sample_count, 64u);
        if (sampling && sample_count && !sample_tokens) return false;
        if (sampling && sample_count) std::memcpy([recent contents], sample_tokens, sample_count * sizeof(int32_t));
        float random_value = 0.0f;
        if (sampling) {
            rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
            random_value = static_cast<float>(rng_state) / static_cast<float>(UINT64_MAX);
        }
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            auto threads = [&](id<MTLComputePipelineState> p, uint32_t count) {
                [encoder setComputePipelineState:p];
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256, count), 1, 1)];
            };
            auto matvec = [&](const mllm::TensorDescriptor* d, id<MTLBuffer> input, id<MTLBuffer> output, uint32_t K, uint32_t N) {
                id<MTLComputePipelineState> pipeline = d->quant_type == 2 ? p_q8 : (d->quant_type == 5 ? p_mq4 : p_q4);
                [encoder setComputePipelineState:pipeline]; [encoder setBuffer:model offset:d->offset atIndex:0];
                [encoder setBuffer:input offset:0 atIndex:1]; [encoder setBuffer:output offset:0 atIndex:2];
                [encoder setBytes:&K length:4 atIndex:3]; [encoder setBytes:&N length:4 atIndex:4];
                const uint32_t groups = d->quant_type == 2 ? N : (N + 1) / 2;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            };
            auto normalize = [&](const mllm::TensorDescriptor* d) {
                const uint32_t dim = header->hidden_dim; const float eps = 1e-5f;
                [encoder setComputePipelineState:p_norm]; [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:model offset:d->offset atIndex:1]; [encoder setBuffer:norm offset:0 atIndex:2];
                [encoder setBytes:&dim length:4 atIndex:3]; [encoder setBytes:&eps length:4 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            };
            auto add = [&](id<MTLBuffer> update) {
                const uint32_t dim = header->hidden_dim; [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:update offset:0 atIndex:1]; [encoder setBytes:&dim length:4 atIndex:2]; threads(p_add, dim);
            };

            const uint32_t dim = header->hidden_dim;
            const size_t row_bytes = embedding->quant_type == 3 ? (dim / 32) * 18 :
                                     (embedding->quant_type == 5 ? (dim / 256) * 144 : dim * 2);
            [encoder setBuffer:model offset:embedding->offset + token * row_bytes atIndex:0];
            [encoder setBuffer:x offset:0 atIndex:1]; [encoder setBytes:&dim length:4 atIndex:2];
            threads(embedding->quant_type == 3 ? p_embedding_q4 :
                    (embedding->quant_type == 5 ? p_embedding_mq4 : p_embedding_fp16), dim);

            for (uint32_t layer_index = 0; layer_index < header->num_layers; ++layer_index) {
                const auto& l = layers[layer_index];
                normalize(l.in_norm);
                matvec(l.q, norm, q, dim, dim); matvec(l.k, norm, k, dim, kv_dim); matvec(l.v, norm, v, dim, kv_dim);
                auto rope = [&](id<MTLBuffer> values, uint32_t heads) {
                    [encoder setComputePipelineState:p_rope]; [encoder setBuffer:values offset:0 atIndex:0];
                    [encoder setBuffer:rope_cos offset:0 atIndex:1]; [encoder setBuffer:rope_sin offset:0 atIndex:2];
                    [encoder setBytes:&head_dim length:4 atIndex:3]; [encoder setBytes:&heads length:4 atIndex:4];
                    [encoder setBytes:&pos length:4 atIndex:5];
                    [encoder dispatchThreads:MTLSizeMake(head_dim / 2, heads, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32, head_dim / 2), 1, 1)];
                };
                rope(q, header->num_heads); rope(k, header->num_kv_heads);
                [encoder setComputePipelineState:q8_kv?p_store_kv_q8:p_store_kv_fp16]; [encoder setBuffer:k offset:0 atIndex:0]; [encoder setBuffer:v offset:0 atIndex:1];
                [encoder setBuffer:cache_k offset:0 atIndex:2]; [encoder setBuffer:cache_v offset:0 atIndex:3];
                [encoder setBytes:&layer_index length:4 atIndex:4]; [encoder setBytes:&pos length:4 atIndex:5];
                [encoder setBytes:&context length:4 atIndex:6]; [encoder setBytes:&header->num_kv_heads length:4 atIndex:7];
                [encoder setBytes:&head_dim length:4 atIndex:8];
                if(q8_kv) [encoder dispatchThreadgroups:MTLSizeMake((head_dim+31)/32,header->num_kv_heads,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                else threads(p_store_kv_fp16,kv_dim);
                [encoder setComputePipelineState:q8_kv?p_attention_q8:p_attention_fp16]; [encoder setBuffer:q offset:0 atIndex:0];
                [encoder setBuffer:cache_k offset:0 atIndex:1]; [encoder setBuffer:cache_v offset:0 atIndex:2]; [encoder setBuffer:attn offset:0 atIndex:3];
                [encoder setBytes:&layer_index length:4 atIndex:4]; [encoder setBytes:&pos length:4 atIndex:5];
                [encoder setBytes:&context length:4 atIndex:6]; [encoder setBytes:&header->num_heads length:4 atIndex:7];
                [encoder setBytes:&header->num_kv_heads length:4 atIndex:8]; [encoder setBytes:&head_dim length:4 atIndex:9];
                [encoder dispatchThreadgroups:MTLSizeMake(header->num_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                matvec(l.o, attn, projection, dim, dim); add(projection); normalize(l.post_norm);
                const uint32_t intermediate = header->intermediate_dim;
                if (l.gate->quant_type == 3 && l.up->quant_type == 3) {
                    [encoder setComputePipelineState:p_swiglu]; [encoder setBuffer:model offset:l.gate->offset atIndex:0];
                    [encoder setBuffer:model offset:l.up->offset atIndex:1]; [encoder setBuffer:norm offset:0 atIndex:2]; [encoder setBuffer:ffn offset:0 atIndex:3];
                    [encoder setBytes:&dim length:4 atIndex:4]; [encoder setBytes:&intermediate length:4 atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake(intermediate, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    matvec(l.gate, norm, gate, dim, intermediate); matvec(l.up, norm, ffn, dim, intermediate);
                    [encoder setBuffer:gate offset:0 atIndex:0]; [encoder setBuffer:ffn offset:0 atIndex:1];
                    [encoder setBytes:&intermediate length:4 atIndex:2]; threads(p_swiglu_element, intermediate);
                }
                matvec(l.down, ffn, projection, intermediate, dim); add(projection);
            }
            if (compute_logits) {
                normalize(final_norm);
                matvec(lm_head, norm, logits, dim, header->vocab_size);
            }
            if (sampling) {
                const uint32_t vocab = header->vocab_size;
                [encoder setComputePipelineState:p_sampler]; [encoder setBuffer:logits offset:0 atIndex:0];
                [encoder setBuffer:recent offset:0 atIndex:1]; [encoder setBuffer:sampled_token offset:0 atIndex:2];
                [encoder setBytes:&vocab length:4 atIndex:3]; [encoder setBytes:&sample_count length:4 atIndex:4];
                [encoder setBytes:&temperature length:4 atIndex:5]; [encoder setBytes:&penalty length:4 atIndex:6];
                [encoder setBytes:&min_p length:4 atIndex:7]; [encoder setBytes:&random_value length:4 atIndex:8];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [encoder endEncoding]; [command commit];
            if (!cpu_logits && !sampling) return true;
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) return false;
            if (cpu_logits) {
                const auto* half_logits = static_cast<const uint16_t*>([logits contents]);
                for (uint32_t i = 0; i < header->vocab_size; ++i) cpu_logits[i] = mllm::fp16_to_fp32(half_logits[i]);
            }
            if (sampling) {
                const uint32_t result = *static_cast<const uint32_t*>([sampled_token contents]);
                *next_token = result < header->vocab_size ? static_cast<int32_t>(result) : 2;
            }
            return true;
        }
    }

    int32_t forward_sample(uint32_t token, uint32_t pos, const int32_t* tokens, uint32_t count,
                           float temperature, float penalty, float min_p) {
        int32_t result = -1;
        return forward(token, pos, true, nullptr, tokens, count, temperature, penalty, min_p, &result)
            ? result : -1;
    }

    int32_t sample(const int32_t* tokens, uint32_t count, float temperature, float penalty, float min_p) {
        count = std::min(count, 64u);
        if (count) std::memcpy([recent contents], tokens, count * sizeof(int32_t));
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
        const float random_value = static_cast<float>(rng_state) / static_cast<float>(UINT64_MAX);
        const uint32_t vocab = header->vocab_size;
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:p_sampler]; [encoder setBuffer:logits offset:0 atIndex:0];
            [encoder setBuffer:recent offset:0 atIndex:1]; [encoder setBuffer:sampled_token offset:0 atIndex:2];
            [encoder setBytes:&vocab length:4 atIndex:3]; [encoder setBytes:&count length:4 atIndex:4];
            [encoder setBytes:&temperature length:4 atIndex:5]; [encoder setBytes:&penalty length:4 atIndex:6];
            [encoder setBytes:&min_p length:4 atIndex:7]; [encoder setBytes:&random_value length:4 atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) return -1;
            const uint32_t result = *static_cast<const uint32_t*>([sampled_token contents]);
            return result < vocab ? static_cast<int32_t>(result) : 2;
        }
    }

    bool prefill(const int32_t* tokens, uint32_t count, uint32_t start_pos) {
        if (!tokens || count < 2 || count > 32 || start_pos + count > context || header->hidden_dim > prefill_hidden_limit || !q8_kv) return false;
        for (const auto& l : layers) if (l.q->quant_type != 3 || l.k->quant_type != 3 || l.v->quant_type != 3 ||
            l.o->quant_type != 3 || l.gate->quant_type != 3 || l.up->quant_type != 3 || l.down->quant_type != 3) return false;
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer]; id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
            const uint32_t dim=header->hidden_dim, intermediate=header->intermediate_dim, qheads=header->num_heads, kvheads=header->num_kv_heads;
            [e setComputePipelineState:embedding->quant_type==3?p_embed_q4_batch:(embedding->quant_type==5?p_embed_mq4_batch:p_embed_batch)];
            [e setBuffer:model offset:embedding->offset atIndex:0]; [e setBytes:tokens length:count*sizeof(int32_t) atIndex:1];
            [e setBuffer:batch_x offset:0 atIndex:2]; [e setBytes:&dim length:4 atIndex:3];
            [e dispatchThreads:MTLSizeMake(dim,count,1) threadsPerThreadgroup:MTLSizeMake(std::min(256u,dim),1,1)];
            auto norm_batch = [&](const mllm::TensorDescriptor* weight) {
                const float eps=1e-5f; [e setComputePipelineState:p_norm_batch]; [e setBuffer:batch_x offset:0 atIndex:0];
                [e setBuffer:model offset:weight->offset atIndex:1]; [e setBuffer:batch_norm offset:0 atIndex:2];
                [e setBytes:&dim length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            };
            auto gemm = [&](const mllm::TensorDescriptor* weight,id<MTLBuffer> input,id<MTLBuffer> output,uint32_t K,uint32_t N) {
                [e setComputePipelineState:p_gemm_batch]; [e setBuffer:model offset:weight->offset atIndex:0]; [e setBuffer:input offset:0 atIndex:1];
                [e setBuffer:output offset:0 atIndex:2]; [e setBytes:&K length:4 atIndex:3]; [e setBytes:&N length:4 atIndex:4]; [e setBytes:&count length:4 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((N+1)/2,(count+3)/4,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            };
            auto add_batch = [&](id<MTLBuffer> update) {
                const uint32_t elements=count*dim; [e setComputePipelineState:p_add_batch]; [e setBuffer:batch_x offset:0 atIndex:0];
                [e setBuffer:update offset:0 atIndex:1]; [e setBytes:&elements length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(elements,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            };
            for(uint32_t layer=0;layer<header->num_layers;++layer){const auto& l=layers[layer]; norm_batch(l.in_norm);
                gemm(l.q,batch_norm,batch_q,dim,dim); gemm(l.k,batch_norm,batch_k,dim,kv_dim); gemm(l.v,batch_norm,batch_v,dim,kv_dim);
                auto rope=[&](id<MTLBuffer> values,uint32_t heads){[e setComputePipelineState:p_rope_batch]; [e setBuffer:values offset:0 atIndex:0];
                    [e setBuffer:rope_cos offset:0 atIndex:1]; [e setBuffer:rope_sin offset:0 atIndex:2]; [e setBytes:&head_dim length:4 atIndex:3];
                    [e setBytes:&heads length:4 atIndex:4]; [e setBytes:&start_pos length:4 atIndex:5];
                    [e dispatchThreads:MTLSizeMake(head_dim/2,heads,count) threadsPerThreadgroup:MTLSizeMake(std::min(32u,head_dim/2),1,1)];};
                rope(batch_q,qheads); rope(batch_k,kvheads);
                [e setComputePipelineState:p_store_batch]; [e setBuffer:batch_k offset:0 atIndex:0]; [e setBuffer:batch_v offset:0 atIndex:1];
                [e setBuffer:cache_k offset:0 atIndex:2]; [e setBuffer:cache_v offset:0 atIndex:3]; [e setBytes:&layer length:4 atIndex:4];
                [e setBytes:&start_pos length:4 atIndex:5]; [e setBytes:&context length:4 atIndex:6]; [e setBytes:&kvheads length:4 atIndex:7]; [e setBytes:&head_dim length:4 atIndex:8];
                [e dispatchThreadgroups:MTLSizeMake((head_dim+31)/32,kvheads,count) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [e setComputePipelineState:p_attention_batch]; [e setBuffer:batch_q offset:0 atIndex:0]; [e setBuffer:cache_k offset:0 atIndex:1];
                [e setBuffer:cache_v offset:0 atIndex:2]; [e setBuffer:batch_attn offset:0 atIndex:3]; [e setBytes:&layer length:4 atIndex:4];
                [e setBytes:&start_pos length:4 atIndex:5]; [e setBytes:&context length:4 atIndex:6]; [e setBytes:&qheads length:4 atIndex:7];
                [e setBytes:&kvheads length:4 atIndex:8]; [e setBytes:&head_dim length:4 atIndex:9];
                [e dispatchThreadgroups:MTLSizeMake(qheads,count,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                gemm(l.o,batch_attn,batch_projection,dim,dim); add_batch(batch_projection); norm_batch(l.post_norm);
                gemm(l.gate,batch_norm,batch_gate,dim,intermediate); gemm(l.up,batch_norm,batch_ffn,dim,intermediate);
                const uint32_t ffn_elements=count*intermediate; [e setComputePipelineState:p_swiglu_batch]; [e setBuffer:batch_gate offset:0 atIndex:0];
                [e setBuffer:batch_ffn offset:0 atIndex:1]; [e setBytes:&ffn_elements length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(ffn_elements,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                gemm(l.down,batch_ffn,batch_projection,intermediate,dim); add_batch(batch_projection);
            }
            norm_batch(final_norm);
            [e setComputePipelineState:lm_head->quant_type==2?p_q8:(lm_head->quant_type==5?p_mq4:p_q4)]; [e setBuffer:model offset:lm_head->offset atIndex:0];
            [e setBuffer:batch_norm offset:(count-1)*dim*sizeof(uint16_t) atIndex:1]; [e setBuffer:logits offset:0 atIndex:2];
            [e setBytes:&dim length:4 atIndex:3]; [e setBytes:&header->vocab_size length:4 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(lm_head->quant_type==2?header->vocab_size:(header->vocab_size+1)/2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [e endEncoding]; [command commit]; return true;
        }
    }

    bool copy_logits(float* output) {
        if (!output) return false;
        id<MTLCommandBuffer> command = [queue commandBuffer]; [command commit]; [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) return false;
        const auto* source = static_cast<const uint16_t*>([logits contents]);
        for (uint32_t i = 0; i < header->vocab_size; ++i) output[i] = mllm::fp16_to_fp32(source[i]);
        return true;
    }
};

extern "C" void* nanoedge_metal_bridge_create(const void* model, size_t length, uint32_t context, float rope_theta) {
    try { return new NanoEdgeMetalBridge(model, length, context, rope_theta); }
    catch (const std::exception& e) { std::cerr << "Metal bridge initialization failed: " << e.what() << '\n'; return nullptr; }
}

extern "C" void nanoedge_metal_bridge_destroy(void* handle) { delete static_cast<NanoEdgeMetalBridge*>(handle); }

extern "C" bool nanoedge_metal_bridge_forward(void* handle, uint32_t token, uint32_t pos, bool logits, float* output) {
    try { return handle && static_cast<NanoEdgeMetalBridge*>(handle)->forward(token, pos, logits, output); }
    catch (...) { return false; }
}

extern "C" int32_t nanoedge_metal_bridge_forward_sample(void* handle, uint32_t token, uint32_t pos,
                                                           const int32_t* recent, uint32_t count,
                                                           float temperature, float penalty, float min_p) {
    try {
        return handle ? static_cast<NanoEdgeMetalBridge*>(handle)->forward_sample(
            token, pos, recent, count, temperature, penalty, min_p) : -1;
    } catch (...) { return -1; }
}

extern "C" int32_t nanoedge_metal_bridge_sample(void* handle, const int32_t* recent, uint32_t count,
                                                  float temperature, float penalty, float min_p) {
    try { return handle ? static_cast<NanoEdgeMetalBridge*>(handle)->sample(recent, count, temperature, penalty, min_p) : -1; }
    catch (...) { return -1; }
}

extern "C" bool nanoedge_metal_bridge_prefill(void* handle, const int32_t* tokens, uint32_t count, uint32_t start_pos) {
    try { return handle && static_cast<NanoEdgeMetalBridge*>(handle)->prefill(tokens,count,start_pos); }
    catch (...) { return false; }
}

extern "C" bool nanoedge_metal_bridge_copy_logits(void* handle, float* output) {
    try { return handle && static_cast<NanoEdgeMetalBridge*>(handle)->copy_logits(output); }
    catch (...) { return false; }
}

extern "C" void nanoedge_metal_bridge_set_prefill_hidden_limit(void* handle, uint32_t limit) {
    if (handle) static_cast<NanoEdgeMetalBridge*>(handle)->prefill_hidden_limit = limit;
}

extern "C" bool nanoedge_metal_bridge_set_kv_precision(void* handle, uint32_t bits) {
    if (!handle || (bits != 8 && bits != 16)) return false;
    try { static_cast<NanoEdgeMetalBridge*>(handle)->configure_kv_cache(bits == 8); return true; }
    catch (...) { return false; }
}

extern "C" size_t nanoedge_metal_bridge_kv_cache_bytes(void* handle) {
    return handle ? static_cast<NanoEdgeMetalBridge*>(handle)->kv_cache_size * 2 : 0;
}
