#pragma once

#include <cstddef>
#include <cstdint>

namespace mllm {

// Quantization format types
enum class QuantType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    Q4_0 = 3,         // 18 bytes: 2-byte fp16 scale + 16-byte packed int4 (32 weights)
    Q4_1 = 4,         // 20 bytes: 2-byte fp16 scale + 2-byte fp16 bias + 16-byte packed int4
    MQ4_APPLE = 5,    // Hardware-tiled: aligned contiguous 4-bit stream + scale block
};

// Execution device target
enum class DeviceType : uint32_t {
    CPU_NEON = 0,
    GPU_METAL = 1,
    NPU_ACCELERATOR = 2,
};

// Transformer layer tensor type identifier
enum class TensorType : uint32_t {
    EMBEDDINGS = 0,
    ATTN_Q = 1,
    ATTN_K = 2,
    ATTN_V = 3,
    ATTN_OUT = 4,
    ATTN_NORM = 5,
    FFN_GATE = 6,
    FFN_UP = 7,
    FFN_DOWN = 8,
    FFN_NORM = 9,
    FINAL_NORM = 10,
    LM_HEAD = 11,
    CUSTOM = 99,
};

// Standard Q4_0 Block: 32 values packed in 18 bytes
#pragma pack(push, 1)
struct BlockQ4_0 {
    uint16_t scale;      // IEEE 754 half-precision float (FP16)
    uint8_t  qs[16];     // 32 4-bit nibbles: [low 4 bits = even, high 4 bits = odd]
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

// Standard Q4_1 Block: 32 values packed in 20 bytes (with min/bias)
#pragma pack(push, 1)
struct BlockQ4_1 {
    uint16_t scale;      // FP16
    uint16_t bias;       // FP16
    uint8_t  qs[16];     // 32 4-bit nibbles
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_1) == 20, "BlockQ4_1 must be 20 bytes");

// MQ4_APPLE Tile: 256 weights packed for SIMD cache-line saturation
// 8 blocks of 32 weights = 256 weights.
// 8 scales (16 bytes) + 128 bytes weights (exactly one 128-byte cache line).
#pragma pack(push, 1)
struct TileMQ4_Apple {
    uint16_t scales[8];  // 16 bytes: 8 FP16 scales
    uint8_t  qs[128];    // 128 bytes: 256 4-bit values (128-byte cache-line aligned)
};
#pragma pack(pop)
static_assert(sizeof(TileMQ4_Apple) == 144, "TileMQ4_Apple must be 144 bytes");

// Standard Q8_0 Block: 32 values packed in 34 bytes (16-bit scale + 32 8-bit ints)
#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t scale;      // IEEE 754 half-precision float (FP16)
    int8_t   qs[32];     // 32 8-bit signed integers
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

// Helper to convert uint16_t bit pattern to float (IEEE 754 half -> float)
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp  = (h >> 10) & 0x001F;
    uint32_t mant = h & 0x03FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t res = (sign << 31);
            return *reinterpret_cast<float*>(&res);
        }
        // Subnormal
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t res = (sign << 31) | 0x7F800000 | (mant << 13);
        return *reinterpret_cast<float*>(&res);
    }

    exp = exp + (127 - 15);
    mant = mant << 13;
    uint32_t res = (sign << 31) | (exp << 23) | mant;
    return *reinterpret_cast<float*>(&res);
}

// Helper to convert float to IEEE 754 half-precision uint16_t
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        return static_cast<uint16_t>(sign << 15);
    } else if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    return static_cast<uint16_t>((sign << 15) | (exp << 10) | (mant >> 13));
}

} // namespace mllm
