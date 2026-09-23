#!/usr/bin/env python3
"""
Convert SmolLM2-135M-Instruct into NanoEdge 16KB page-aligned Q4_0 .mllm format
with real trained weights and full 30-layer causal transformer topology.
"""

import os
import sys
import json
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MLLM_MAGIC = 0x4D4C4C4D  # "MLLM"
MLLM_VERSION = 1
MLLM_PAGE_ALIGNMENT = 16384  # 16 KB

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
    Quantize [rows, cols] float32 array to BlockQ4_0 format.
    BlockQ4_0: 2 bytes FP16 scale + 16 bytes (32 packed 4-bit nibbles) = 18 bytes.
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
    scales_bytes = scales.tobytes()
    packed_bytes = packed.tobytes()
    
    out_buf = bytearray(num_blocks * 18)
    for i in range(num_blocks):
        out_buf[i*18 : i*18 + 2] = scales_bytes[i*2 : (i+1)*2]
        out_buf[i*18 + 2 : (i+1)*18] = packed_bytes[i*16 : (i+1)*16]
        
    return bytes(out_buf)

def quantize_q8_0(weights_fp32: np.ndarray) -> bytes:
    """
    Quantize [rows, cols] float32 array to BlockQ8_0 format.
    BlockQ8_0: 2 bytes FP16 scale + 32 bytes int8 = 34 bytes.
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
    
    scales_bytes = scales.tobytes()
    quant_bytes = quantized.tobytes()
    
    out_buf = bytearray(num_blocks * 34)
    for i in range(num_blocks):
        out_buf[i*34 : i*34 + 2] = scales_bytes[i*2 : (i+1)*2]
        out_buf[i*34 + 2 : (i+1)*34] = quant_bytes[i*32 : (i+1)*32]
        
    return bytes(out_buf)

def main():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[NanoEdge Compiler] Loading '{model_id}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    intermediate_dim = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    vocab_size = cfg.vocab_size
    
    print(f"[NanoEdge Compiler] Model Config: {num_layers} layers, dim={hidden_dim}, ffn_dim={intermediate_dim}, heads={num_heads}/{num_kv_heads}, vocab={vocab_size}")
    
    output_dir = "models/suite"
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "smollm2_135m_q4.mllm")
    output_vocab_path = os.path.join(output_dir, "smollm2_vocab.json")
    
    # 1. Export Vocab & Tokenizer mappings
    print(f"[NanoEdge Compiler] Exporting Tokenizer Vocab to {output_vocab_path}...")
    vocab = tokenizer.get_vocab()
    id_to_token = [""] * vocab_size
    for token, idx in vocab.items():
        if idx < vocab_size:
            id_to_token[idx] = token
            
    tokenizer_data = {
        "vocab_size": vocab_size,
        "bos_token_id": tokenizer.bos_token_id or 1,
        "eos_token_id": tokenizer.eos_token_id or 2,
        "pad_token_id": tokenizer.pad_token_id or 2,
        "tokens": id_to_token
    }
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
    print(f"[NanoEdge Compiler] Vocab exported ({len(id_to_token)} tokens).")
    
    # 2. Extract & Quantize Tensors
    tensors_to_write = []
    
    # Embeddings (FP16)
    print("[NanoEdge Compiler] Processing Embeddings...")
    embed_weights = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float16)
    tensors_to_write.append({
        "name": "model.embed_tokens.weight",
        "type": TENSOR_TYPES["embed_tokens"],
        "layer_idx": -1,
        "quant_type": QUANT_TYPES["FP16"],
        "rows": embed_weights.shape[0],
        "cols": embed_weights.shape[1],
        "data": embed_weights.tobytes(),
    })
    
    # 30 Transformer Layers
    for l in range(num_layers):
        if (l + 1) % 5 == 0 or l == 0:
            print(f"[NanoEdge Compiler] Quantizing Layer {l+1}/{num_layers}...")
        layer = model.model.layers[l]
        
        # Input LayerNorm (FP16)
        in_norm = layer.input_layernorm.weight.detach().cpu().numpy().astype(np.float16)
        tensors_to_write.append({
            "name": f"model.layers.{l}.input_layernorm.weight",
            "type": TENSOR_TYPES["attn_norm"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["FP16"],
            "rows": 1,
            "cols": in_norm.shape[0],
            "data": in_norm.tobytes(),
        })
        
        # Q, K, V Projections (Q4_0)
        q_w = layer.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.q_proj.weight",
            "type": TENSOR_TYPES["attn_q"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": q_w.shape[0],
            "cols": q_w.shape[1],
            "data": quantize_q4_0(q_w),
        })
        
        k_w = layer.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.k_proj.weight",
            "type": TENSOR_TYPES["attn_k"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": k_w.shape[0],
            "cols": k_w.shape[1],
            "data": quantize_q4_0(k_w),
        })
        
        v_w = layer.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.v_proj.weight",
            "type": TENSOR_TYPES["attn_v"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": v_w.shape[0],
            "cols": v_w.shape[1],
            "data": quantize_q4_0(v_w),
        })
        
        o_w = layer.self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.self_attn.o_proj.weight",
            "type": TENSOR_TYPES["attn_out"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": o_w.shape[0],
            "cols": o_w.shape[1],
            "data": quantize_q4_0(o_w),
        })
        
        # Post Attention LayerNorm (FP16)
        post_norm = layer.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float16)
        tensors_to_write.append({
            "name": f"model.layers.{l}.post_attention_layernorm.weight",
            "type": TENSOR_TYPES["ffn_norm"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["FP16"],
            "rows": 1,
            "cols": post_norm.shape[0],
            "data": post_norm.tobytes(),
        })
        
        # Gate, Up, Down Projections (Q4_0)
        gate_w = layer.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.mlp.gate_proj.weight",
            "type": TENSOR_TYPES["ffn_gate"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": gate_w.shape[0],
            "cols": gate_w.shape[1],
            "data": quantize_q4_0(gate_w),
        })
        
        up_w = layer.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.mlp.up_proj.weight",
            "type": TENSOR_TYPES["ffn_up"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": up_w.shape[0],
            "cols": up_w.shape[1],
            "data": quantize_q4_0(up_w),
        })
        
        down_w = layer.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
        tensors_to_write.append({
            "name": f"model.layers.{l}.mlp.down_proj.weight",
            "type": TENSOR_TYPES["ffn_down"],
            "layer_idx": l,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": down_w.shape[0],
            "cols": down_w.shape[1],
            "data": quantize_q4_0(down_w),
        })
        
    # Final LayerNorm (FP16)
    final_norm = model.model.norm.weight.detach().cpu().numpy().astype(np.float16)
    tensors_to_write.append({
        "name": "model.norm.weight",
        "type": TENSOR_TYPES["final_norm"],
        "layer_idx": -1,
        "quant_type": QUANT_TYPES["FP16"],
        "rows": 1,
        "cols": final_norm.shape[0],
        "data": final_norm.tobytes(),
    })
    
    # LM Head (Q8_0 INT8 for high-precision vocabulary projection)
    print("[NanoEdge Compiler] Quantizing LM Head (Q8_0 INT8)...")
    lm_w = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    tensors_to_write.append({
        "name": "lm_head.weight",
        "type": TENSOR_TYPES["lm_head"],
        "layer_idx": -1,
        "quant_type": QUANT_TYPES["Q8_0"],
        "rows": lm_w.shape[0],
        "cols": lm_w.shape[1],
        "data": quantize_q8_0(lm_w),
    })
    
    # 3. Layout & Page Align Binary Container
    print("[NanoEdge Compiler] Assembling 16KB page-aligned binary container...")
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
        1, # Architecture: Llama/Qwen/SmolLM
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        num_kv_heads,
        vocab_size,
        2048, # max_seq_len
        MLLM_PAGE_ALIGNMENT,
        num_tensors,
        manifest_offset,
        manifest_size,
        weights_offset,
        total_file_size,
        b'\x00' * 48
    )
    assert len(header_bytes) == 128, f"Header size mismatch: {len(header_bytes)}"
    
    with open(output_model_path, "wb") as f:
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
    print(f"[NanoEdge Compiler] SUCCESS! Generated '{output_model_path}' ({file_mb:.2f} MB, {num_tensors} tensors).")
    
    # Copy to iOS App bundle directory as well
    ios_target_dir = "mobile/ios/NanoEdgeApp"
    os.system(f"cp '{output_model_path}' '{ios_target_dir}/'")
    os.system(f"cp '{output_vocab_path}' '{ios_target_dir}/'")
    print(f"[NanoEdge Compiler] Copied binary & vocab to '{ios_target_dir}/'.")

if __name__ == "__main__":
    main()
