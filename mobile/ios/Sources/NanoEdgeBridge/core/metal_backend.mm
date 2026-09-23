#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

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

kernel void q4_0_gemv(
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
)metal";

struct MetalBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLLibrary> library{nil};

    id<MTLComputePipelineState> pso_q4_0_gemv{nil};
    id<MTLComputePipelineState> pso_mq4_apple_gemv{nil};
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

        // Collect device metadata
        pimpl_->device_info.device_name = std::string([[pimpl_->device name] UTF8String]);
        pimpl_->device_info.has_unified_memory = [pimpl_->device hasUnifiedMemory];
        pimpl_->device_info.max_buffer_length = [pimpl_->device maxBufferLength];
        pimpl_->device_info.max_threads_per_threadgroup = 1024;

        // Compile Shader Library
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:EMBEDDED_METAL_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop

        pimpl_->library = [pimpl_->device newLibraryWithSource:source options:options error:&error];
        if (!pimpl_->library) {
            std::string err_msg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            throw std::runtime_error("Metal shader compilation failed: " + err_msg);
        }

        // Create Compute Pipeline States
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
        pimpl_->pso_mq4_apple_gemv = create_pso(@"mq4_apple_gemv");
        pimpl_->pso_rms_norm = create_pso(@"rms_norm");
        pimpl_->pso_rope = create_pso(@"rope_embedding");
    }
}

MetalBackend::~MetalBackend() {
    // ARC handles Objective-C releases
}

const DeviceInfo& MetalBackend::device_info() const noexcept {
    return pimpl_->device_info;
}

void* MetalBackend::create_buffer_no_copy(const void* bytes, size_t length) {
    @autoreleasepool {
        // Round length up to 16KB page boundary required by Apple Silicon Metal
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

        // 128 threads per threadgroup (4 SIMD-groups of 32 threads)
        MTLSize threads_per_tg = MTLSizeMake(128, 1, 1);
        MTLSize grid_size = MTLSizeMake(N, 1, 1); // 1 threadgroup per row

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
