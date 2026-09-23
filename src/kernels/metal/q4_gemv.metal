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
