#import "include/NanoEdgeBridge.h"
#include "mllm/types.h"
#include "mllm/model_format.h"
#include "runtime/memory_mapped_model.hpp"
#include "kernels/metal/metal_backend.hpp"
#include "kernels/cpu/neon_gemv.hpp"

#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

@implementation NanoEdgeMemoryStats
@end

@implementation NanoEdgeBenchmarkResult
@end

@interface NanoEdgeBridge () {
    std::unique_ptr<mllm::MemoryMappedModel> _model;
    std::unique_ptr<mllm::MetalBackend> _metal;
}
@end

@implementation NanoEdgeBridge

+ (instancetype)sharedInstance {
    static NanoEdgeBridge* instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[NanoEdgeBridge alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        try {
            _metal = std::make_unique<mllm::MetalBackend>();
        } catch (const std::exception& e) {
            NSLog(@"[NanoEdgeBridge] Metal initialization error: %s", e.what());
        }
    }
    return self;
}

- (NSString*)deviceName {
    if (_metal) {
        return [NSString stringWithUTF8String:_metal->device_info().device_name.c_str()];
    }
    return @"Unknown Device";
}

- (BOOL)hasUnifiedMemory {
    if (_metal) {
        return _metal->device_info().has_unified_memory ? YES : NO;
    }
    return NO;
}

- (NanoEdgeMemoryStats*)queryMemoryFootprint {
    NanoEdgeMemoryStats* stats = [[NanoEdgeMemoryStats alloc] init];
    auto s = mllm::MemoryMappedModel::query_process_memory_stats();
    stats.virtualSizeMB = s.virtual_size_bytes / (1024.0 * 1024.0);
    stats.residentSizeMB = s.resident_size_bytes / (1024.0 * 1024.0);
    stats.physicalFootprintMB = s.dirty_size_bytes / (1024.0 * 1024.0);
    return stats;
}

- (BOOL)loadModelFromPath:(NSString*)filePath error:(NSError**)error {
    try {
        _model = std::make_unique<mllm::MemoryMappedModel>([filePath UTF8String]);
        return YES;
    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:@"NanoEdgeBridge"
                                         code:1001
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
        }
        return NO;
    }
}

- (BOOL)isModelLoaded {
    return _model != nullptr;
}

- (NSString*)loadedModelInfo {
    if (!_model) return @"No model loaded";
    const auto& hdr = _model->header();
    double mb = _model->total_mapped_bytes() / (1024.0 * 1024.0);
    return [NSString stringWithFormat:@"%d Layers | Dim: %d | FFN: %d | Size: %.1f MB",
            hdr.num_layers, hdr.hidden_dim, hdr.intermediate_dim, mb];
}

- (NanoEdgeBenchmarkResult*)runDecodeBenchmark:(NSInteger)iterations {
    if (!_model || !_metal) return nil;

    // Pick target projection (Q proj from layer 0)
    const mllm::TensorDescriptor* target = nullptr;
    for (const auto& d : _model->descriptors()) {
        if (d.layer_idx == 0 && (d.tensor_type == static_cast<uint32_t>(mllm::TensorType::ATTN_Q) ||
                                 d.tensor_type == static_cast<uint32_t>(mllm::TensorType::FFN_GATE))) {
            target = &d;
            break;
        }
    }
    if (!target) target = &_model->descriptors()[0];

    uint32_t N = target->rows;
    uint32_t K = target->cols;
    size_t weight_bytes = target->size_bytes;
    const void* raw_weights = _model->get_tensor_data(*target);

    // Zero-copy Metal buffer binding
    void* mtl_weights = _metal->create_buffer_no_copy(raw_weights, weight_bytes);
    void* mtl_input = _metal->allocate_buffer(K * sizeof(uint16_t));
    void* mtl_output = _metal->allocate_buffer(N * sizeof(uint16_t));

    // Prepare normalized input vector
    std::vector<float> input_f32(K);
    std::vector<uint16_t> input_fp16(K);
    for (uint32_t i = 0; i < K; ++i) {
        input_f32[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.1f;
        input_fp16[i] = mllm::fp32_to_fp16(input_f32[i]);
    }
    std::memcpy(_metal->get_buffer_contents(mtl_input), input_fp16.data(), K * sizeof(uint16_t));

    // CPU baseline validation
    std::vector<float> cpu_out(N, 0.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (target->quant_type == static_cast<uint32_t>(mllm::QuantType::Q4_0)) {
        mllm::NeonGemv::compute_q4_0(static_cast<const mllm::BlockQ4_0*>(raw_weights), input_f32.data(), cpu_out.data(), K, N);
    } else {
        mllm::NeonGemv::compute_mq4_apple(static_cast<const mllm::TileMQ4_Apple*>(raw_weights), input_f32.data(), cpu_out.data(), K, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_lat_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // GPU validation run
    if (target->quant_type == static_cast<uint32_t>(mllm::QuantType::Q4_0)) {
        _metal->dispatch_q4_0_gemv(mtl_weights, mtl_input, mtl_output, K, N);
    } else {
        _metal->dispatch_mq4_apple_gemv(mtl_weights, mtl_input, mtl_output, K, N);
    }

    // Verify error
    std::vector<float> gpu_out(N);
    const auto* gpu_fp16 = static_cast<const uint16_t*>(_metal->get_buffer_contents(mtl_output));
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        gpu_out[i] = mllm::fp16_to_fp32(gpu_fp16[i]);
        max_diff = std::max(max_diff, std::abs(cpu_out[i] - gpu_out[i]));
    }

    // Benchmark loop
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (NSInteger i = 0; i < iterations; ++i) {
        double lat = _metal->dispatch_q4_0_gemv(mtl_weights, mtl_input, mtl_output, K, N);
        latencies.push_back(lat);
    }
    std::sort(latencies.begin(), latencies.end());

    double min_lat = latencies.front();
    double med_lat = latencies[iterations / 2];

    double total_bytes = weight_bytes + (K * sizeof(uint16_t)) + (N * sizeof(uint16_t));
    double effective_bw = (total_bytes / (med_lat * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    // Approximate token decode ceiling assuming ~7 projections per layer
    uint32_t layers = _model->header().num_layers;
    if (layers == 0) layers = 1;
    double approx_time_per_token_s = (med_lat * 1e-6) * (layers * 7.0);
    double tok_per_sec = approx_time_per_token_s > 0 ? (1.0 / approx_time_per_token_s) : 0.0;

    _metal->release_buffer(mtl_weights);
    _metal->release_buffer(mtl_input);
    _metal->release_buffer(mtl_output);

    NanoEdgeBenchmarkResult* res = [[NanoEdgeBenchmarkResult alloc] init];
    res.tensorName = [NSString stringWithUTF8String:target->name];
    res.rows = N;
    res.cols = K;
    res.weightSizeMB = weight_bytes / (1024.0 * 1024.0);
    res.cpuLatencyUs = cpu_lat_us;
    res.gpuMinLatencyUs = min_lat;
    res.gpuMedianLatencyUs = med_lat;
    res.effectiveBandwidthGBs = effective_bw;
    res.tokensPerSecCeiling = tok_per_sec;
    res.validationPassed = (max_diff < 0.05f);

    return res;
}

- (void)handleMemoryWarning {
    if (_model) {
        NSLog(@"[NanoEdgeBridge] iOS Memory Warning received! Evicting cold layers via madvise...");
        for (int32_t l = 0; l < static_cast<int32_t>(_model->header().num_layers); ++l) {
            _model->evict_layer(l);
        }
    }
}

@end
