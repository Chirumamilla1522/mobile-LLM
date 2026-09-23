#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>

namespace mllm {

// 34-byte INT8 KV block for 32 elements (1.06 bytes per element vs 2.0 bytes for FP16)
#pragma pack(push, 1)
struct BlockKVQ8 {
    uint16_t scale;    // FP16 scale factor
    int8_t values[32]; // 32 8-bit quantized elements
};
#pragma pack(pop)
static_assert(sizeof(BlockKVQ8) == 34, "BlockKVQ8 must be 34 bytes");

class QuantizedKVCache {
public:
    QuantizedKVCache(uint32_t num_layers, uint32_t num_kv_heads, uint32_t head_dim, uint32_t max_seq_len)
        : num_layers_(num_layers),
          num_kv_heads_(num_kv_heads),
          head_dim_(head_dim),
          max_seq_len_(max_seq_len),
          blocks_per_vector_((head_dim + 31) / 32),
          current_len_(0) {
        
        size_t total_blocks_per_layer = static_cast<size_t>(num_kv_heads_) * max_seq_len_ * blocks_per_vector_;
        k_blocks_.resize(num_layers_ * total_blocks_per_layer);
        v_blocks_.resize(num_layers_ * total_blocks_per_layer);
    }

    // Memory footprint calculation
    [[nodiscard]] size_t memory_bytes() const noexcept {
        return (k_blocks_.size() + v_blocks_.size()) * sizeof(BlockKVQ8);
    }

    [[nodiscard]] double memory_mb() const noexcept {
        return static_cast<double>(memory_bytes()) / (1024.0 * 1024.0);
    }

    [[nodiscard]] size_t fp16_equivalent_bytes() const noexcept {
        return static_cast<size_t>(num_layers_) * 2 * num_kv_heads_ * max_seq_len_ * head_dim_ * sizeof(uint16_t);
    }

    [[nodiscard]] double fp16_equivalent_mb() const noexcept {
        return static_cast<double>(fp16_equivalent_bytes()) / (1024.0 * 1024.0);
    }

    [[nodiscard]] double memory_saved_mb() const noexcept {
        return fp16_equivalent_mb() - memory_mb();
    }

    [[nodiscard]] double compression_ratio() const noexcept {
        return memory_bytes() > 0 ? (static_cast<double>(fp16_equivalent_bytes()) / memory_bytes()) : 1.0;
    }

    [[nodiscard]] uint32_t current_len() const noexcept { return current_len_; }
    [[nodiscard]] uint32_t max_seq_len() const noexcept { return max_seq_len_; }

    void advance_step() noexcept {
        if (current_len_ < max_seq_len_) current_len_++;
    }

    void reset() noexcept {
        current_len_ = 0;
    }

private:
    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    uint32_t blocks_per_vector_;
    uint32_t current_len_;

    std::vector<BlockKVQ8> k_blocks_;
    std::vector<BlockKVQ8> v_blocks_;
};

} // namespace mllm
