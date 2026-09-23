#pragma once

#include "runtime/memory_mapped_model.hpp"
#include "mllm/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace mllm {

struct GenerationConfig {
    float temperature = 0.2f;
    float repetition_penalty = 1.15f;
    float min_p = 0.05f;
    int max_tokens = 256;
};

class TransformerEngine {
public:
    explicit TransformerEngine(std::shared_ptr<MemoryMappedModel> model);
    ~TransformerEngine();

    bool load_vocab(const std::string& vocab_json_path);
    
    std::vector<int32_t> encode_prompt(const std::string& user_prompt, const std::string& system_prompt);
    std::string decode_token(int32_t token_id);

    // Run full autoregressive decode generation
    void generate(
        const std::string& user_prompt,
        const std::string& system_prompt,
        const GenerationConfig& config,
        std::function<bool(const std::string& token, double tok_per_sec)> on_token,
        std::function<void(const std::string& full_text, double total_time_s, double avg_tok_s)> on_complete
    );

    void generate_from_tokens(
        const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& config,
        std::function<bool(const std::string& token, double tok_per_sec)> on_token,
        std::function<void(const std::string& full_text, double total_time_s, double avg_tok_s)> on_complete
    );

    bool is_ready() const { return model_ != nullptr && vocab_loaded_; }
    size_t vocab_size() const { return vocab_.size(); }

private:
    std::shared_ptr<MemoryMappedModel> model_;
    bool vocab_loaded_ = false;

    // Vocab mappings
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    int32_t bos_token_id_ = 1;
    int32_t eos_token_id_ = 2;

    // Model Architecture Hyperparameters
    uint32_t num_layers_ = 30;
    uint32_t hidden_dim_ = 576;
    uint32_t intermediate_dim_ = 1536;
    uint32_t num_heads_ = 9;
    uint32_t num_kv_heads_ = 3;
    uint32_t head_dim_ = 64;
    float rms_norm_eps_ = 1e-5f;
    float rope_theta_ = 100000.0f;

    // Tensor weight pointers mapped directly from mmap
    const uint16_t* embed_tokens_ = nullptr; // FP16
    const uint16_t* final_norm_ = nullptr;   // FP16
    const void* lm_head_ = nullptr;          // Q4_0 or Q8_0
    uint32_t lm_head_quant_type_ = 3;        // Default Q4_0

    struct LayerWeights {
        const uint16_t* in_norm = nullptr;   // FP16
        const void* q_proj = nullptr;        // Q4_0
        const void* k_proj = nullptr;        // Q4_0
        const void* v_proj = nullptr;        // Q4_0
        const void* o_proj = nullptr;        // Q4_0
        const uint16_t* post_norm = nullptr; // FP16
        const void* gate_proj = nullptr;     // Q4_0
        const void* up_proj = nullptr;       // Q4_0
        const void* down_proj = nullptr;     // Q4_0
    };
    std::vector<LayerWeights> layers_;

    // Static Pre-Allocated Activation Buffers (Zero Heap Allocations during Decode)
    std::vector<float> x_;
    std::vector<float> x_norm_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> attn_out_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_;
    std::vector<float> logits_;

    // KV Cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
    static constexpr size_t MAX_SEQ_LEN = 1024;
    std::vector<float> kv_cache_k_;
    std::vector<float> kv_cache_v_;

    void bind_tensors();
    void forward_token(int32_t token_id, int32_t pos);
    int32_t sample_next_token(const GenerationConfig& config, const std::vector<int32_t>& recent_tokens);
    void apply_rms_norm(const float* in, const uint16_t* weight, float* out, size_t dim);
    void apply_rope(float* vec, size_t num_heads, size_t head_dim, int32_t pos);
};

} // namespace mllm
