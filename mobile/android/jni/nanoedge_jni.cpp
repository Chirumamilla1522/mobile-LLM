#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "mllm/types.h"
#include "mllm/model_format.h"
#include "runtime/memory_mapped_model.hpp"
#include "kernels/cpu/neon_gemv.hpp"

// Global model instance for Android runtime
static std::unique_ptr<mllm::MemoryMappedModel> g_android_model;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_nanoedge_mobile_NanoEdgeEngine_nativeLoadModel(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath
) {
    const char* path_chars = env->GetStringUTFChars(modelPath, nullptr);
    if (!path_chars) return JNI_FALSE;

    try {
        g_android_model = std::make_unique<mllm::MemoryMappedModel>(path_chars);
        env->ReleaseStringUTFChars(modelPath, path_chars);
        return JNI_TRUE;
    } catch (...) {
        env->ReleaseStringUTFChars(modelPath, path_chars);
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL
Java_com_nanoedge_mobile_NanoEdgeEngine_nativeGetModelInfo(
    JNIEnv* env,
    jobject /* this */
) {
    if (!g_android_model) {
        return env->NewStringUTF("No model loaded");
    }
    const auto& hdr = g_android_model->header();
    double mb = g_android_model->total_mapped_bytes() / (1024.0 * 1024.0);
    char buf[256];
    snprintf(buf, sizeof(buf), "Layers: %u | Dim: %u | FFN: %u | Size: %.1f MB",
             hdr.num_layers, hdr.hidden_dim, hdr.intermediate_dim, mb);
    return env->NewStringUTF(buf);
}

JNIEXPORT jdoubleArray JNICALL
Java_com_nanoedge_mobile_NanoEdgeEngine_nativeRunBenchmark(
    JNIEnv* env,
    jobject /* this */,
    jint iterations
) {
    if (!g_android_model) return nullptr;

    const mllm::TensorDescriptor* target = nullptr;
    for (const auto& d : g_android_model->descriptors()) {
        if (d.layer_idx == 0 && (d.tensor_type == static_cast<uint32_t>(mllm::TensorType::ATTN_Q) ||
                                 d.tensor_type == static_cast<uint32_t>(mllm::TensorType::FFN_GATE))) {
            target = &d;
            break;
        }
    }
    if (!target) target = &g_android_model->descriptors()[0];

    uint32_t N = target->rows;
    uint32_t K = target->cols;
    const void* raw_weights = g_android_model->get_tensor_data(*target);

    std::vector<float> input_f32(K, 0.05f);
    std::vector<float> output_f32(N, 0.0f);

    // Warmup
    mllm::NeonGemv::compute_q4_0(
        static_cast<const mllm::BlockQ4_0*>(raw_weights),
        input_f32.data(),
        output_f32.data(),
        K, N
    );

    auto t0 = std::chrono::high_resolution_clock::now();
    for (jint i = 0; i < iterations; ++i) {
        mllm::NeonGemv::compute_q4_0(
            static_cast<const mllm::BlockQ4_0*>(raw_weights),
            input_f32.data(),
            output_f32.data(),
            K, N
        );
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = total_us / iterations;

    // Calculate effective memory bandwidth in GB/s
    double weight_bytes = target->size_bytes;
    double bw_gbs = (weight_bytes / (avg_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    jdoubleArray result = env->NewDoubleArray(3);
    jdouble vals[3] = { avg_us, bw_gbs, static_cast<double>(N) };
    env->SetDoubleArrayRegion(result, 0, 3, vals);
    return result;
}

} // extern "C"
