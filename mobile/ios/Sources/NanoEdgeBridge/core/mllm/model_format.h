#pragma once

#include "mllm/types.h"
#include <cstdint>
#include <string_view>

namespace mllm {

constexpr uint32_t MLLM_MAGIC = 0x4D4C4C4D; // "MLLM" in ASCII
constexpr uint32_t MLLM_VERSION = 1;
constexpr uint32_t MLLM_PAGE_ALIGNMENT = 16384; // 16 KB (Apple Silicon native page size)

// Binary Model Header (128 bytes, aligned)
#pragma pack(push, 1)
struct ModelHeader {
    uint32_t magic;                 // MLLM_MAGIC
    uint32_t version;               // MLLM_VERSION
    uint32_t architecture;          // 0: Generic, 1: Llama/Qwen, 2: Gemma
    uint32_t num_layers;            // e.g. 28 for Qwen-3B, 32 for Llama-8B
    uint32_t hidden_dim;            // e.g. 2048 / 4096
    uint32_t intermediate_dim;      // e.g. 11008 / 14336 (FFN hidden size)
    uint32_t num_heads;             // Attention heads
    uint32_t num_kv_heads;          // KV heads for GQA (Grouped Query Attention)
    uint32_t vocab_size;            // e.g. 32000, 151936
    uint32_t max_seq_len;           // e.g. 4096, 32768
    uint32_t page_size;             // 16384 bytes
    uint32_t num_tensors;           // Total tensors in descriptor table
    uint64_t manifest_offset;       // Byte offset to TensorDescriptor table
    uint64_t manifest_size;         // Total bytes of descriptor table
    uint64_t weights_offset;        // Byte offset where aligned layer weights start
    uint64_t total_file_size;       // Total size of .mllm file
    uint8_t  reserved[48];          // Reserved for future extensions (pad to 128 bytes)
};
#pragma pack(pop)
static_assert(sizeof(ModelHeader) == 128, "ModelHeader must be exactly 128 bytes");

// Tensor Descriptor within the manifest (128 bytes, aligned)
#pragma pack(push, 1)
struct TensorDescriptor {
    char        name[64];           // Canonical tensor name, null-terminated
    uint32_t    tensor_type;        // TensorType enum
    int32_t     layer_idx;          // -1 for global (embed, head), 0..num_layers-1 for layers
    uint32_t    quant_type;         // QuantType enum (Q4_0, MQ4_APPLE, FP16, etc.)
    uint32_t    rows;               // N dimension
    uint32_t    cols;               // K dimension
    uint64_t    offset;             // Absolute byte offset in file (16KB aligned)
    uint64_t    size_bytes;         // Total byte size of tensor data
    uint64_t    scales_offset;      // Secondary offset for separated scales (or 0 if interleaved)
    uint32_t    scales_bytes;       // Secondary size for scales (or 0)
    uint8_t     reserved[16];       // Pad to 128 bytes (64 + 5*4 + 3*8 + 4 + 16 = 128)
};
#pragma pack(pop)
static_assert(sizeof(TensorDescriptor) == 128, "TensorDescriptor must be exactly 128 bytes");

// Helper to align an offset to the next page boundary
inline uint64_t align_to_page(uint64_t offset, uint64_t page_size = MLLM_PAGE_ALIGNMENT) {
    return (offset + page_size - 1) & ~(page_size - 1);
}

} // namespace mllm
