#!/usr/bin/env python3
"""
MLLM Model Compiler & Packer (NanoEdge)
Converts FP32/FP16 weights or generates synthetic benchmarks into sequential,
16KB page-aligned, INT4-quantized .mllm container binaries for mobile unified memory.
"""

import argparse
import struct
import numpy as np
import os
import sys

MLLM_MAGIC = 0x4D4C4C4D  # "MLLM"
MLLM_VERSION = 1
MLLM_PAGE_ALIGNMENT = 16384  # 16 KB native Apple Silicon page size

# TensorType identifiers
TENSOR_TYPES = {
    "embed_tokens": 0,
    "attn_q": 1,
    "attn_k": 2,
    "attn_v": 3,
    "attn_out": 4,
    "attn_norm": 5,
    "ffn_gate": 6,
    "ffn_up": 7,
    "ffn_down": 8,
    "ffn_norm": 9,
    "final_norm": 10,
    "lm_head": 11,
}

QUANT_TYPES = {
    "FP32": 0,
    "FP16": 1,
    "INT8": 2,
    "Q8_0": 2,
    "Q4_0": 3,
    "Q4_1": 4,
    "MQ4_APPLE": 5,
}

def align_to_page(offset: int, page_size: int = MLLM_PAGE_ALIGNMENT) -> int:
    return (offset + page_size - 1) & ~(page_size - 1)

def quantize_q4_0(weights_fp32: np.ndarray) -> bytes:
    """
    Quantize 2D matrix [rows, cols] to BlockQ4_0 format.
    cols must be a multiple of 32.
    Each block: 2 bytes FP16 scale + 16 bytes packed int4 nibbles = 18 bytes.
    """
    rows, cols = weights_fp32.shape
    assert cols % 32 == 0, f"cols ({cols}) must be divisible by 32"
    
    num_blocks = (rows * cols) // 32
    reshaped = weights_fp32.reshape(-1, 32)
    
    # Calculate scale per block
    max_vals = np.max(np.abs(reshaped), axis=1)
    scales = (max_vals / 7.0).astype(np.float16)
    scales[scales == 0] = np.float16(1e-5)
    
    # Quantize to [-8, 7] and offset by +8 -> [0, 15]
    scales_f32 = scales.astype(np.float32)[:, np.newaxis]
    quantized = np.clip(np.round(reshaped / scales_f32), -8, 7).astype(np.int8) + 8
    quantized = quantized.astype(np.uint8)
    
    # Pack low nibble = even index, high nibble = odd index
    even_vals = quantized[:, 0::2]
    odd_vals = quantized[:, 1::2]
    packed = (even_vals & 0x0F) | ((odd_vals & 0x0F) << 4)
    
    # Interleave scale (2 bytes) + packed (16 bytes)
    out_buf = bytearray()
    scales_bytes = scales.tobytes()
    packed_bytes = packed.tobytes()
    
    for i in range(num_blocks):
        out_buf.extend(scales_bytes[i*2:(i+1)*2])
        out_buf.extend(packed_bytes[i*16:(i+1)*16])
        
    return bytes(out_buf)

def quantize_mq4_apple(weights_fp32: np.ndarray) -> bytes:
    """
    Quantize 2D matrix [rows, cols] to TileMQ4_Apple format.
    cols must be a multiple of 256.
    Each tile: 8 scales (16 bytes) + 128 bytes packed int4 nibbles = 144 bytes.
    """
    rows, cols = weights_fp32.shape
    assert cols % 256 == 0, f"cols ({cols}) must be divisible by 256"
    
    num_tiles = (rows * cols) // 256
    reshaped_blocks = weights_fp32.reshape(-1, 32)
    
    max_vals = np.max(np.abs(reshaped_blocks), axis=1)
    scales = (max_vals / 7.0).astype(np.float16)
    scales[scales == 0] = np.float16(1e-5)
    
    scales_f32 = scales.astype(np.float32)[:, np.newaxis]
    quantized = np.clip(np.round(reshaped_blocks / scales_f32), -8, 7).astype(np.int8) + 8
    quantized = quantized.astype(np.uint8)
    
    even_vals = quantized[:, 0::2]
    odd_vals = quantized[:, 1::2]
    packed_blocks = (even_vals & 0x0F) | ((odd_vals & 0x0F) << 4) # [num_blocks, 16]
    
    out_buf = bytearray()
    scales_bytes = scales.tobytes()
    packed_bytes = packed_blocks.tobytes()
    
    for t in range(num_tiles):
        # 8 scales = 16 bytes
        out_buf.extend(scales_bytes[t*16:(t+1)*16])
        # 8 blocks * 16 bytes = 128 bytes
        out_buf.extend(packed_bytes[t*128:(t+1)*128])
        
    return bytes(out_buf)

def quantize_q8_0(weights_fp32: np.ndarray) -> bytes:
    """
    Quantize 2D matrix [rows, cols] to BlockQ8_0 format.
    cols must be a multiple of 32.
    Each block: 2 bytes FP16 scale + 32 bytes int8 values = 34 bytes.
    """
    rows, cols = weights_fp32.shape
    assert cols % 32 == 0, f"cols ({cols}) must be divisible by 32"
    
    num_blocks = (rows * cols) // 32
    reshaped = weights_fp32.reshape(-1, 32)
    
    max_vals = np.max(np.abs(reshaped), axis=1)
    scales = (max_vals / 127.0).astype(np.float16)
    scales[scales == 0] = np.float16(1e-5)
    
    scales_f32 = scales.astype(np.float32)[:, np.newaxis]
    quantized = np.clip(np.round(reshaped / scales_f32), -128, 127).astype(np.int8)
    
    out_buf = bytearray()
    scales_bytes = scales.tobytes()
    quant_bytes = quantized.tobytes()
    
    for i in range(num_blocks):
        out_buf.extend(scales_bytes[i*2:(i+1)*2])
        out_buf.extend(quant_bytes[i*32:(i+1)*32])
        
    return bytes(out_buf)

def quantize_tensor(weights_fp32: np.ndarray, quant_type: str) -> bytes:
    if quant_type == "Q4_0":
        return quantize_q4_0(weights_fp32)
    elif quant_type == "MQ4_APPLE":
        return quantize_mq4_apple(weights_fp32)
    elif quant_type in ("INT8", "Q8_0"):
        return quantize_q8_0(weights_fp32)
    else:
        return weights_fp32.astype(np.float16).tobytes()

def pack_synthetic_model(output_path: str, hidden_dim: int, intermediate_dim: int, num_layers: int, quant_type: str, vocab_size: int = 1000, num_heads: int = 32, num_kv_heads: int = 8):
    print(f"[Compiler] Generating model: {num_layers} layers, dim={hidden_dim}, ffn_dim={intermediate_dim}, heads={num_heads}/{num_kv_heads}, quant={quant_type}, vocab={vocab_size}")
    
    quant_code = QUANT_TYPES[quant_type]
    tensors_to_write = []
    
    # 1. Embeddings (FP16)
    embed_data = (np.random.randn(vocab_size, hidden_dim) * 0.02).astype(np.float16).tobytes()
    tensors_to_write.append({
        "name": "model.embed_tokens.weight",
        "type": TENSOR_TYPES["embed_tokens"],
        "layer_idx": -1,
        "quant_type": QUANT_TYPES["FP16"],
        "rows": vocab_size,
        "cols": hidden_dim,
        "data": embed_data,
    })
    
    # 2. Sequential Layers
    for l in range(num_layers):
        # Input Norm (FP16)
        norm_data = np.ones((hidden_dim,), dtype=np.float16).tobytes()
        tensors_to_write.append({
            "name": f"model.layers.{l}.input_layernorm.weight",
            "type": TENSOR_TYPES["attn_norm"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["FP16"],
            "rows": 1,
            "cols": hidden_dim,
            "data": norm_data,
        })
        
        # Attention Q, K, V
        # Q: [hidden_dim, hidden_dim]
        w_q = (np.random.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float32)
        qdata_q = quantize_tensor(w_q, quant_type)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.q_proj.weight",
            "type": TENSOR_TYPES["attn_q"],
            "layer_idx": l,
            "quant_type": quant_code,
            "rows": hidden_dim,
            "cols": hidden_dim,
            "data": qdata_q,
        })
        
        # K, V: [kv_dim, hidden_dim] where kv_dim = (num_kv_heads * head_dim)
        head_dim = hidden_dim // max(num_heads, 1)
        kv_dim = num_kv_heads * head_dim
        # Ensure divisible by 32
        if kv_dim % 32 != 0:
            kv_dim = ((kv_dim + 31) // 32) * 32
        
        for proj in ["k", "v"]:
            w_kv = (np.random.randn(kv_dim, hidden_dim) * 0.02).astype(np.float32)
            qdata_kv = quantize_tensor(w_kv, quant_type)
            tensors_to_write.append({
                "name": f"model.layers.{l}.self_attn.{proj}_proj.weight",
                "type": TENSOR_TYPES[f"attn_{proj}"],
                "layer_idx": l,
                "quant_type": quant_code,
                "rows": kv_dim,
                "cols": hidden_dim,
                "data": qdata_kv,
            })
            
        # Attention Output [hidden_dim, hidden_dim]
        w_out = (np.random.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float32)
        qdata_out = quantize_tensor(w_out, quant_type)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.o_proj.weight",
            "type": TENSOR_TYPES["attn_out"],
            "layer_idx": l,
            "quant_type": quant_code,
            "rows": hidden_dim,
            "cols": hidden_dim,
            "data": qdata_out,
        })
        
        # Post Attention Norm (FP16)
        tensors_to_write.append({
            "name": f"model.layers.{l}.post_attention_layernorm.weight",
            "type": TENSOR_TYPES["ffn_norm"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["FP16"],
            "rows": 1,
            "cols": hidden_dim,
            "data": norm_data,
        })
        
        # FFN Gate and Up [intermediate_dim, hidden_dim]
        for ffn in ["gate", "up"]:
            w_ffn = (np.random.randn(intermediate_dim, hidden_dim) * 0.02).astype(np.float32)
            qdata_ffn = quantize_tensor(w_ffn, quant_type)
            tensors_to_write.append({
                "name": f"model.layers.{l}.mlp.{ffn}_proj.weight",
                "type": TENSOR_TYPES[f"ffn_{ffn}"],
                "layer_idx": l,
                "quant_type": quant_code,
                "rows": intermediate_dim,
                "cols": hidden_dim,
                "data": qdata_ffn,
            })
            
        # FFN Down [hidden_dim, intermediate_dim]
        w_down = (np.random.randn(hidden_dim, intermediate_dim) * 0.02).astype(np.float32)
        qdata_down = quantize_tensor(w_down, quant_type)
        tensors_to_write.append({
            "name": f"model.layers.{l}.mlp.down_proj.weight",
            "type": TENSOR_TYPES["ffn_down"],
            "layer_idx": l,
            "quant_type": quant_code,
            "rows": hidden_dim,
            "cols": intermediate_dim,
            "data": qdata_down,
        })
        
    # Final Norm & LM Head
    final_norm_data = np.ones((hidden_dim,), dtype=np.float16).tobytes()
    tensors_to_write.append({
        "name": "model.norm.weight",
        "type": TENSOR_TYPES["final_norm"],
        "layer_idx": -1,
        "quant_type": QUANT_TYPES["FP16"],
        "rows": 1,
        "cols": hidden_dim,
        "data": final_norm_data,
    })
    
    lm_head = (np.random.randn(vocab_size, hidden_dim) * 0.02).astype(np.float32)
    qdata_lm = quantize_tensor(lm_head, quant_type)
    tensors_to_write.append({
        "name": "lm_head.weight",
        "type": TENSOR_TYPES["lm_head"],
        "layer_idx": -1,
        "quant_type": quant_code,
        "rows": vocab_size,
        "cols": hidden_dim,
        "data": qdata_lm,
    })
    
    # Calculate layouts and page alignments
    num_tensors = len(tensors_to_write)
    header_size = 128
    manifest_offset = header_size
    manifest_size = num_tensors * 128
    
    current_offset = align_to_page(manifest_offset + manifest_size, MLLM_PAGE_ALIGNMENT)
    weights_offset = current_offset
    
    descriptors = []
    tensor_payloads = []
    
    for t in tensors_to_write:
        data_len = len(t["data"])
        tensor_offset = current_offset
        
        name_bytes = t["name"].encode("utf-8")[:63].ljust(64, b'\x00')
        desc_bytes = struct.pack(
            "<64sIiIIIQQQI16s",
            name_bytes,
            t["type"],
            t["layer_idx"],
            t["quant_type"],
            t["rows"],
            t["cols"],
            tensor_offset,
            data_len,
            0, # scales_offset
            0, # scales_bytes
            b'\x00' * 16
        )
        assert len(desc_bytes) == 128, f"Descriptor size mismatch: {len(desc_bytes)}"
        descriptors.append(desc_bytes)
        tensor_payloads.append(t["data"])
        
        current_offset = align_to_page(current_offset + data_len, MLLM_PAGE_ALIGNMENT)
        
    total_file_size = current_offset
    
    header_bytes = struct.pack(
        "<IIIIIIIIIIIIQQQQ48s",
        MLLM_MAGIC,
        MLLM_VERSION,
        1, # Architecture: Llama/Qwen/Mistral
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        num_kv_heads,
        vocab_size,
        4096, # max_seq_len
        MLLM_PAGE_ALIGNMENT,
        num_tensors,
        manifest_offset,
        manifest_size,
        weights_offset,
        total_file_size,
        b'\x00' * 48
    )
    assert len(header_bytes) == 128, f"Header size mismatch: {len(header_bytes)}"
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header_bytes)
        for desc in descriptors:
            f.write(desc)
            
        current_pos = f.tell()
        if current_pos < weights_offset:
            f.write(b'\x00' * (weights_offset - current_pos))
            
        for i, payload in enumerate(tensor_payloads):
            target_offset = struct.unpack("<Q", descriptors[i][84:92])[0]
            current_pos = f.tell()
            if current_pos < target_offset:
                f.write(b'\x00' * (target_offset - current_pos))
            f.write(payload)
            
        final_pos = f.tell()
        if final_pos < total_file_size:
            f.write(b'\x00' * (total_file_size - final_pos))
            
    file_mb = total_file_size / (1024 * 1024)
    print(f"[Compiler] Successfully generated '{output_path}' ({file_mb:.2f} MB, {num_tensors} tensors, page-aligned).")

# Official Open-Source Mobile LLM Architecture Specifications
MODEL_PRESETS = {
    # --- QWEN 2.5 FAMILY ---
    "qwen2.5_0.5b": {
        "hidden_dim": 896,
        "intermediate_dim": 4864,
        "num_layers": 2, # benchmark layers
        "num_heads": 14,
        "num_kv_heads": 2,
        "vocab_size": 1000,
        "family": "Qwen",
        "description": "Alibaba Qwen-2.5-0.5B Mobile Optimized"
    },
    "qwen2.5_1.5b": {
        "hidden_dim": 1536,
        "intermediate_dim": 8960,
        "num_layers": 2,
        "num_heads": 12,
        "num_kv_heads": 2,
        "vocab_size": 1000,
        "family": "Qwen",
        "description": "Alibaba Qwen-2.5-1.5B High Performance"
    },
    "qwen2.5_3b": {
        "hidden_dim": 2048,
        "intermediate_dim": 11008,
        "num_layers": 2,
        "num_heads": 16,
        "num_kv_heads": 2,
        "vocab_size": 1000,
        "family": "Qwen",
        "description": "Alibaba Qwen-2.5-3B Reasoning Tier"
    },
    # --- LLAMA 3.2 / 3.1 FAMILY ---
    "llama3.2_1b": {
        "hidden_dim": 2048,
        "intermediate_dim": 8192,
        "num_layers": 2,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 1000,
        "family": "Llama",
        "description": "Meta LLaMA-3.2-1B On-Device Edge"
    },
    "llama3.2_3b": {
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "num_layers": 2,
        "num_heads": 24,
        "num_kv_heads": 8,
        "vocab_size": 1000,
        "family": "Llama",
        "description": "Meta LLaMA-3.2-3B Multilingual Edge"
    },
    "llama3.1_8b": {
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 2,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 1000,
        "family": "Llama",
        "description": "Meta LLaMA-3.1-8B Desktop & Pro Tier"
    },
    # --- MISTRAL FAMILY ---
    "mistral_7b": {
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 2,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 1000,
        "family": "Mistral",
        "description": "Mistral-7B-v0.3 Dense Transformer"
    },
    # --- GEMMA 2 FAMILY ---
    "gemma2_2b": {
        "hidden_dim": 2304,
        "intermediate_dim": 9216,
        "num_layers": 2,
        "num_heads": 8,
        "num_kv_heads": 4,
        "vocab_size": 1000,
        "family": "Gemma",
        "description": "Google Gemma-2-2B Compact Architecture"
    },
}

def build_suite(suite_dir: str = "models/suite"):
    os.makedirs(suite_dir, exist_ok=True)
    
    suite_configs = [
        # --- 1. QWEN FAMILY MODELS ---
        ("qwen2.5_0.5b_q4_0.mllm", 896, 4864, 2, "Q4_0", 14, 2),
        ("qwen2.5_1.5b_q4_0.mllm", 1536, 8960, 2, "Q4_0", 12, 2),
        ("qwen2.5_1.5b_mq4.mllm", 1536, 8960, 2, "MQ4_APPLE", 12, 2),
        ("qwen2.5_1.5b_int8.mllm", 1536, 8960, 2, "INT8", 12, 2),
        ("qwen2.5_3b_q4_0.mllm", 2048, 11008, 2, "Q4_0", 16, 2),

        # --- 2. LLAMA FAMILY MODELS ---
        ("llama3.2_1b_q4_0.mllm", 2048, 8192, 2, "Q4_0", 32, 8),
        ("llama3.2_1b_mq4.mllm", 2048, 8192, 2, "MQ4_APPLE", 32, 8),
        ("llama3.2_3b_q4_0.mllm", 3072, 8192, 2, "Q4_0", 24, 8),
        ("llama3.2_3b_mq4.mllm", 3072, 8192, 2, "MQ4_APPLE", 24, 8),

        # --- 3. MISTRAL FAMILY MODELS ---
        ("mistral_7b_q4_0.mllm", 4096, 14336, 2, "Q4_0", 32, 8),
        ("mistral_7b_mq4.mllm", 4096, 14336, 2, "MQ4_APPLE", 32, 8),

        # --- 4. GEMMA FAMILY MODELS ---
        ("gemma2_2b_q4_0.mllm", 2304, 9216, 2, "Q4_0", 8, 4),

        # --- 5. SCALE-TIER SUITE REFERENCE MODELS ---
        ("nano_q4_0.mllm", 1024, 2816, 2, "Q4_0", 32, 8),
        ("nano_mq4_apple.mllm", 1024, 2816, 2, "MQ4_APPLE", 32, 8),
        ("nano_int8.mllm", 1024, 2816, 2, "INT8", 32, 8),
        ("small_q4_0.mllm", 2048, 5632, 2, "Q4_0", 32, 8),
        ("small_mq4_apple.mllm", 2048, 5632, 2, "MQ4_APPLE", 32, 8),
        ("small_int8.mllm", 2048, 5632, 2, "INT8", 32, 8),
        ("medium_q4_0.mllm", 3072, 8192, 2, "Q4_0", 32, 8),
        ("medium_mq4_apple.mllm", 3072, 8192, 2, "MQ4_APPLE", 32, 8),
        ("large_q4_0.mllm", 4096, 14336, 2, "Q4_0", 32, 8),
        ("large_mq4_apple.mllm", 4096, 14336, 2, "MQ4_APPLE", 32, 8),
    ]

    print(f"=== Building Mobile LLM Benchmark Suite ({len(suite_configs)} Models) in '{suite_dir}' ===")
    for filename, dim, ffn, layers, quant, heads, kv_heads in suite_configs:
        out_path = os.path.join(suite_dir, filename)
        pack_synthetic_model(out_path, dim, ffn, layers, quant, vocab_size=1000, num_heads=heads, num_kv_heads=kv_heads)
    print("=== Suite Generation Complete ===")

def main():
    parser = argparse.ArgumentParser(description="MLLM Model Compiler & Packer for Mobile LLMs (Qwen, Llama, Mistral, Gemma)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test model")
    parser.add_argument("--preset", type=str, choices=list(MODEL_PRESETS.keys()), help="Model preset (e.g. qwen2.5_0.5b, llama3.2_1b, mistral_7b)")
    parser.add_argument("--build-suite", action="store_true", help="Generate complete multi-family benchmark suite")
    parser.add_argument("--suite-dir", type=str, default="models/suite", help="Directory for benchmark suite")
    parser.add_argument("--dim", type=int, default=2048, help="Hidden dimension (K)")
    parser.add_argument("--hidden-dim", type=int, default=5632, help="Intermediate FFN dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads (GQA)")
    parser.add_argument("--vocab", type=int, default=1000, help="Vocab size")
    parser.add_argument("--quant", type=str, default="Q4_0", choices=["Q4_0", "MQ4_APPLE", "INT8", "Q8_0"], help="Quantization format")
    parser.add_argument("--out", type=str, default="models/test_model.mllm", help="Output .mllm path")
    args = parser.parse_args()
    
    if args.build_suite:
        build_suite(args.suite_dir)
    elif args.preset:
        spec = MODEL_PRESETS[args.preset]
        out_path = args.out if args.out != "models/test_model.mllm" else f"models/{args.preset}_{args.quant.lower()}.mllm"
        pack_synthetic_model(
            out_path,
            spec["hidden_dim"],
            spec["intermediate_dim"],
            args.layers if args.layers else spec["num_layers"],
            args.quant,
            spec["vocab_size"],
            spec["num_heads"],
            spec["num_kv_heads"]
        )
    elif args.synthetic:
        pack_synthetic_model(args.out, args.dim, args.hidden_dim, args.layers, args.quant, args.vocab, args.heads, args.kv_heads)
    else:
        print("Please provide --build-suite, --preset, or --synthetic.")

if __name__ == "__main__":
    main()
