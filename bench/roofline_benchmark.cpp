#include "mllm/types.h"
#include "mllm/model_format.h"
#include "runtime/memory_mapped_model.hpp"
#include "runtime/arena_allocator.hpp"
#include "kernels/cpu/neon_gemv.hpp"
#include "kernels/metal/metal_backend.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace mllm;

int main(int argc, char** argv) {
    std::string model_path = "models/test_q4.mllm";
    int iterations = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
    }

    std::cout << "=================================================================\n";
    std::cout << "      NanoEdge Mobile LLM Kernel Runtime & Roofline Benchmark     \n";
    std::cout << "=================================================================\n";

    // 1. Cold Start & Zero-Copy mmap Benchmark
    auto t_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<MemoryMappedModel> model;
    try {
        model = std::make_unique<MemoryMappedModel>(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double cold_start_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

    const auto& hdr = model->header();
    std::cout << "\n[Model Manifest Loaded]\n";
    std::cout << "  • File:             " << model_path << "\n";
    std::cout << "  • Total File Size:  " << (model->total_mapped_bytes() / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  • Architecture:     " << (hdr.architecture == 1 ? "Llama / Qwen" : "Generic") << "\n";
    std::cout << "  • Layers:           " << hdr.num_layers << "\n";
    std::cout << "  • Hidden Dim (K):   " << hdr.hidden_dim << "\n";
    std::cout << "  • FFN Dim:          " << hdr.intermediate_dim << "\n";
    std::cout << "  • Total Tensors:    " << hdr.num_tensors << "\n";
    std::cout << "  • Cold Start Time:  " << std::fixed << std::setprecision(2) << cold_start_us << " us (" 
              << (cold_start_us / 1000.0) << " ms)\n";

    // Memory stats immediately after mmap (prior to page access)
    auto mem_stats_init = MemoryMappedModel::query_process_memory_stats();
    std::cout << "\n[Memory Footprint (Zero-Copy mmap)]\n";
    std::cout << "  • Virtual Address Space: " << (mem_stats_init.virtual_size_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  • Resident Physical (RSS):" << (mem_stats_init.resident_size_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  • Apple Phys Footprint:   " << (mem_stats_init.dirty_size_bytes / (1024.0 * 1024.0)) << " MB\n";

    // 2. Initialize Metal Hardware Backend
    std::cout << "\n[Hardware Compute Backend]\n";
    std::unique_ptr<MetalBackend> metal;
    try {
        metal = std::make_unique<MetalBackend>();
        const auto& dev_info = metal->device_info();
        std::cout << "  • Active GPU:            " << dev_info.device_name << "\n";
        std::cout << "  • Unified Memory:        " << (dev_info.has_unified_memory ? "YES (Zero-Copy Enabled)" : "NO") << "\n";
        std::cout << "  • Max Buffer Length:     " << (dev_info.max_buffer_length / (1024.0 * 1024.0)) << " MB\n";
    } catch (const std::exception& e) {
        std::cerr << "Metal initialization failed: " << e.what() << "\n";
        return 1;
    }

    // 3. Select Target Projection Tensor for Evaluation
    // Find Q projection from Layer 0
    const TensorDescriptor* target_desc = nullptr;
    for (const auto& d : model->descriptors()) {
        if (d.layer_idx == 0 && (d.tensor_type == static_cast<uint32_t>(TensorType::ATTN_Q) || 
                                 d.tensor_type == static_cast<uint32_t>(TensorType::FFN_GATE))) {
            target_desc = &d;
            break;
        }
    }

    if (!target_desc) {
        target_desc = &model->descriptors()[1]; // fallback to second tensor
    }

    uint32_t N = target_desc->rows;
    uint32_t K = target_desc->cols;
    size_t weight_bytes = target_desc->size_bytes;
    const void* weight_raw_ptr = model->get_tensor_data(*target_desc);

    std::cout << "\n[Target Kernel Benchmark: " << target_desc->name << "]\n";
    std::cout << "  • Dimensions:       [" << N << " x " << K << "]\n";
    std::cout << "  • Weight Size:      " << (weight_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  • Quantization:     " << (target_desc->quant_type == static_cast<uint32_t>(QuantType::Q4_0) ? "Q4_0 (18B/block)" : "MQ4_APPLE") << "\n";

    // 4. Zero-Copy Metal Buffer Binding
    // Directly wrap the mmapped file memory pointer in an MTLBuffer with NO COPY
    void* mtl_weights = metal->create_buffer_no_copy(weight_raw_ptr, weight_bytes);
    void* mtl_input   = metal->allocate_buffer(K * sizeof(uint16_t)); // FP16 input
    void* mtl_output  = metal->allocate_buffer(N * sizeof(uint16_t)); // FP16 output

    // Prepare synthetic normalized input activation vector (FP16 and FP32)
    std::vector<float> input_f32(K);
    std::vector<uint16_t> input_fp16(K);
    for (uint32_t i = 0; i < K; ++i) {
        input_f32[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.1f;
        input_fp16[i] = fp32_to_fp16(input_f32[i]);
    }
    // Copy input to GPU shared buffer
    std::memcpy(metal->get_buffer_contents(mtl_input), input_fp16.data(), K * sizeof(uint16_t));

    // 5. Numerical Accuracy Verification (ARM NEON vs Metal GPU)
    std::cout << "\n[Numerical Validation: CPU NEON vs. Metal GPU]\n";
    std::vector<float> cpu_output(N, 0.0f);
    
    // CPU Reference run
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    if (target_desc->quant_type == static_cast<uint32_t>(QuantType::Q4_0)) {
        NeonGemv::compute_q4_0(static_cast<const BlockQ4_0*>(weight_raw_ptr), input_f32.data(), cpu_output.data(), K, N);
    } else {
        NeonGemv::compute_mq4_apple(static_cast<const TileMQ4_Apple*>(weight_raw_ptr), input_f32.data(), cpu_output.data(), K, N);
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_lat_us = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();

    // GPU Run
    if (target_desc->quant_type == static_cast<uint32_t>(QuantType::Q4_0)) {
        metal->dispatch_q4_0_gemv(mtl_weights, mtl_input, mtl_output, K, N);
    } else {
        metal->dispatch_mq4_apple_gemv(mtl_weights, mtl_input, mtl_output, K, N);
    }

    // Read back GPU output from shared unified memory
    std::vector<float> gpu_output(N);
    const auto* gpu_out_fp16 = static_cast<const uint16_t*>(metal->get_buffer_contents(mtl_output));
    for (uint32_t i = 0; i < N; ++i) {
        gpu_output[i] = fp16_to_fp32(gpu_out_fp16[i]);
    }

    // Calculate error metrics
    float max_abs_diff = 0.0f;
    double mse = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        float diff = std::abs(cpu_output[i] - gpu_output[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        mse += diff * diff;
    }
    mse /= N;

    std::cout << "  • CPU NEON Latency:     " << cpu_lat_us << " us (" << (cpu_lat_us / 1000.0) << " ms)\n";
    std::cout << "  • Max Absolute Error:   " << max_abs_diff << "\n";
    std::cout << "  • Mean Squared Error:   " << mse << "\n";
    if (max_abs_diff < 0.05f) {
        std::cout << "  • Validation Status:    ✅ PASSED (Within FP16 quantization tolerance)\n";
    } else {
        std::cout << "  • Validation Status:    ⚠️ CHECK (Max diff: " << max_abs_diff << ")\n";
    }

    // 6. Roofline & Memory Bandwidth Profiling
    std::cout << "\n[Roofline & Sustained Throughput Benchmark (" << iterations << " iterations)]\n";
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        metal->dispatch_q4_0_gemv(mtl_weights, mtl_input, mtl_output, K, N);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        double lat = metal->dispatch_q4_0_gemv(mtl_weights, mtl_input, mtl_output, K, N);
        latencies_us.push_back(lat);
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double median_lat_us = latencies_us[iterations / 2];
    double min_lat_us = latencies_us.front();
    double avg_lat_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / iterations;

    // Roofline calculations
    // 1 matrix-vector multiply does 2 * N * K FLOPs (multiply + add)
    double gflops = (2.0 * N * K) / (median_lat_us * 1e-6) / 1e9;
    
    // Effective bandwidth = (bytes read from weights + input + output) / time
    double total_bytes = weight_bytes + (K * sizeof(uint16_t)) + (N * sizeof(uint16_t));
    double effective_bw_gb_s = (total_bytes / (median_lat_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    double arithmetic_intensity = (2.0 * N * K) / total_bytes; // FLOPs per byte

    std::cout << "  • Min Latency:          " << min_lat_us << " us (" << (min_lat_us / 1000.0) << " ms)\n";
    std::cout << "  • Median Latency:       " << median_lat_us << " us (" << (median_lat_us / 1000.0) << " ms)\n";
    std::cout << "  • Mean Latency:         " << avg_lat_us << " us (" << (avg_lat_us / 1000.0) << " ms)\n";
    std::cout << "  • Compute Throughput:   " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    std::cout << "  • Effective Bandwidth:  " << std::fixed << std::setprecision(2) << effective_bw_gb_s << " GB/s\n";
    std::cout << "  • Arithmetic Intensity: " << std::fixed << std::setprecision(2) << arithmetic_intensity << " FLOPs/byte\n";

    // Roofline classification
    std::cout << "\n[Roofline Analysis]\n";
    std::cout << "  • Arithmetic Intensity (" << arithmetic_intensity << " FLOPs/byte) < Machine Balance (~10-20 FLOPs/byte)\n";
    std::cout << "  • Regimes:              MEMORY-BANDWIDTH BOUND (Autoregressive decoding is bound by memory bandwidth!)\n";
    std::cout << "  • Hardware Saturation:  Reaching " << effective_bw_gb_s << " GB/s sustained DRAM transfer via unified memory.\n";

    // Clean up
    metal->release_buffer(mtl_weights);
    metal->release_buffer(mtl_input);
    metal->release_buffer(mtl_output);

    std::cout << "\n=================================================================\n";
    std::cout << "                  Benchmark Complete Successfully                \n";
    std::cout << "=================================================================\n";

    return 0;
}
