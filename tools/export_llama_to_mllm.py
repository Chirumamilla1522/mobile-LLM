#!/usr/bin/env python3
"""
Convert cached LLaMA 3.2 1B & 3B Instruct 4-bit models into NanoEdge
16 KB page-aligned BlockQ4_0 .mllm format with real weights and zero-duplication
tied LM head projections.
"""

import os
import sys
import glob
import json
import struct
import argparse
import time
import numpy as np
from safetensors import safe_open
import mlx.core as mx

MLLM_MAGIC = 0x4D4C4C4D  # "MLLM"
MLLM_VERSION = 1
MLLM_PAGE_ALIGNMENT = 16384  # 16 KB (Apple Silicon page size)

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

def quantize_q4_0_fast(weights_fp32: np.ndarray) -> bytes:
    """
    Quantize [rows, cols] float32 array to BlockQ4_0 format.
    BlockQ4_0: 2 bytes FP16 scale + 16 bytes (32 packed 4-bit nibbles) = 18 bytes.
    """
    rows, cols = weights_fp32.shape
    assert cols % 32 == 0, f"cols ({cols}) must be divisible by 32"

    reshaped = weights_fp32.reshape(-1, 32)
    max_vals = np.max(np.abs(reshaped), axis=1)
    scales = (max_vals / 7.0).astype(np.float16)
    scales[scales == 0] = np.float16(1e-5)

    scales_f32 = scales.astype(np.float32)[:, np.newaxis]
    quant = np.clip(np.round(reshaped / scales_f32), -8, 7).astype(np.int8) + 8
    quant = quant.astype(np.uint8)

    even = quant[:, 0::2]
    odd = quant[:, 1::2]
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)

    dt = np.dtype([('scale', '<f2'), ('qs', '16u1')])
    blocks = np.empty(len(scales), dtype=dt)
    blocks['scale'] = scales
    blocks['qs'] = packed
    return blocks.tobytes()

def dequantize_mlx_weight(st_file, base_key: str, group_size: int = 64) -> np.ndarray:
    w = mx.array(st_file.get_tensor(f"{base_key}.weight"))
    s = mx.array(st_file.get_tensor(f"{base_key}.scales"))
    b = mx.array(st_file.get_tensor(f"{base_key}.biases"))
    deq = mx.dequantize(w, s, b, group_size=group_size, bits=4)
    mx.eval(deq)
    return np.array(deq, dtype=np.float32)

def export_vocab(cache_dir: str, output_path: str):
    tok_json_path = os.path.join(cache_dir, "tokenizer.json")
    print(f"[Exporter] Exporting tokenizer vocabulary from {tok_json_path}...")
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data["model"]["vocab"]
    added = tok_data.get("added_tokens", [])

    max_id = max(max(vocab.values()), max([x["id"] for x in added]) if added else 0)
    vocab_size = max_id + 1

    tokens = [""] * vocab_size
    for token, idx in vocab.items():
        if idx < vocab_size:
            tokens[idx] = token
    for at in added:
        idx = at["id"]
        if idx < vocab_size:
            tokens[idx] = at["content"]

    export_data = {
        "vocab_size": vocab_size,
        "bos_token_id": 128000, # <|begin_of_text|>
        "eos_token_id": 128009, # <|eot_id|>
        "tokens": tokens
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False)

    print(f"[Exporter] ✅ Successfully exported {vocab_size} tokens to {output_path}")

def convert_model(model_tag: str, cache_pattern: str, output_mllm_path: str):
    snapshots = glob.glob(cache_pattern)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot found matching: {cache_pattern}")
    model_dir = snapshots[0]
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")

    print(f"\n==================================================")
    print(f"[Exporter] Starting conversion for {model_tag}")
    print(f"  Safetensors: {safetensors_path}")
    print(f"  Config:      {config_path}")
    print(f"==================================================")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    num_layers = cfg["num_hidden_layers"]
    hidden_dim = cfg["hidden_size"]
    intermediate_dim = cfg["intermediate_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    vocab_size = cfg["vocab_size"]
    group_size = cfg.get("quantization", {}).get("group_size", 64)

    print(f"[Exporter] Config: {num_layers} layers, dim={hidden_dim}, ffn={intermediate_dim}, heads={num_heads}/{num_kv_heads}, vocab={vocab_size}")

    t_start = time.time()
    tensors_to_write = []

    with safe_open(safetensors_path, framework="numpy") as st:
        # 1. Embeddings (Q4_0)
        print("[Exporter] Processing embed_tokens (Q4_0)...")
        t0 = time.time()
        embed_fp32 = dequantize_mlx_weight(st, "model.embed_tokens", group_size=group_size)
        embed_q4_bytes = quantize_q4_0_fast(embed_fp32)
        print(f"  Processed embed_tokens in {time.time()-t0:.2f}s, size: {len(embed_q4_bytes)/(1024*1024):.1f} MB")

        embed_tensor_dict = {
            "name": "model.embed_tokens.weight",
            "type": TENSOR_TYPES["embed_tokens"],
            "layer_idx": -1,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": embed_fp32.shape[0],
            "cols": embed_fp32.shape[1],
            "data": embed_q4_bytes,
            "shared_with": None
        }
        tensors_to_write.append(embed_tensor_dict)

        # 2. Transformer Layers
        for l in range(num_layers):
            layer_t0 = time.time()
            if l % 4 == 0 or l == num_layers - 1:
                print(f"[Exporter] Processing Layer {l+1}/{num_layers}...")

            # Input Layernorm (FP16)
            in_norm = st.get_tensor(f"model.layers.{l}.input_layernorm.weight").astype(np.float16)
            tensors_to_write.append({
                "name": f"model.layers.{l}.input_layernorm.weight",
                "type": TENSOR_TYPES["attn_norm"],
                "layer_idx": l,
                "quant_type": QUANT_TYPES["FP16"],
                "rows": 1,
                "cols": in_norm.shape[0],
                "data": in_norm.tobytes(),
                "shared_with": None
            })

            # Q, K, V, O projections (Q4_0)
            for proj_name, ttype in [
                ("q_proj", TENSOR_TYPES["attn_q"]),
                ("k_proj", TENSOR_TYPES["attn_k"]),
                ("v_proj", TENSOR_TYPES["attn_v"]),
                ("o_proj", TENSOR_TYPES["attn_out"]),
            ]:
                base_k = f"model.layers.{l}.self_attn.{proj_name}"
                w_f32 = dequantize_mlx_weight(st, base_k, group_size=group_size)
                q4_bytes = quantize_q4_0_fast(w_f32)
                tensors_to_write.append({
                    "name": f"{base_k}.weight",
                    "type": ttype,
                    "layer_idx": l,
                    "quant_type": QUANT_TYPES["Q4_0"],
                    "rows": w_f32.shape[0],
                    "cols": w_f32.shape[1],
                    "data": q4_bytes,
                    "shared_with": None
                })

            # Post attention layernorm (FP16)
            post_norm = st.get_tensor(f"model.layers.{l}.post_attention_layernorm.weight").astype(np.float16)
            tensors_to_write.append({
                "name": f"model.layers.{l}.post_attention_layernorm.weight",
                "type": TENSOR_TYPES["ffn_norm"],
                "layer_idx": l,
                "quant_type": QUANT_TYPES["FP16"],
                "rows": 1,
                "cols": post_norm.shape[0],
                "data": post_norm.tobytes(),
                "shared_with": None
            })

            # Gate, Up, Down projections (Q4_0)
            for proj_name, ttype in [
                ("gate_proj", TENSOR_TYPES["ffn_gate"]),
                ("up_proj", TENSOR_TYPES["ffn_up"]),
                ("down_proj", TENSOR_TYPES["ffn_down"]),
            ]:
                base_k = f"model.layers.{l}.mlp.{proj_name}"
                w_f32 = dequantize_mlx_weight(st, base_k, group_size=group_size)
                q4_bytes = quantize_q4_0_fast(w_f32)
                tensors_to_write.append({
                    "name": f"{base_k}.weight",
                    "type": ttype,
                    "layer_idx": l,
                    "quant_type": QUANT_TYPES["Q4_0"],
                    "rows": w_f32.shape[0],
                    "cols": w_f32.shape[1],
                    "data": q4_bytes,
                    "shared_with": None
                })

        # 3. Final LayerNorm (FP16)
        print("[Exporter] Processing final norm...")
        final_norm = st.get_tensor("model.norm.weight").astype(np.float16)
        tensors_to_write.append({
            "name": "model.norm.weight",
            "type": TENSOR_TYPES["final_norm"],
            "layer_idx": -1,
            "quant_type": QUANT_TYPES["FP16"],
            "rows": 1,
            "cols": final_norm.shape[0],
            "data": final_norm.tobytes(),
            "shared_with": None
        })

        # 4. LM Head (Tied to embed_tokens - 0 duplicate bytes!)
        print("[Exporter] Registering tied lm_head (sharing embed_tokens offset)...")
        tensors_to_write.append({
            "name": "lm_head.weight",
            "type": TENSOR_TYPES["lm_head"],
            "layer_idx": -1,
            "quant_type": QUANT_TYPES["Q4_0"],
            "rows": embed_tensor_dict["rows"],
            "cols": embed_tensor_dict["cols"],
            "data": b"", # Shares payload with embed_tokens
            "shared_with": "model.embed_tokens.weight"
        })

    # Assemble 16 KB Page-Aligned Container
    print(f"\n[Exporter] Assembling 16 KB page-aligned binary file...")
    num_tensors = len(tensors_to_write)
    header_size = 128
    manifest_offset = header_size
    manifest_size = num_tensors * 128

    current_offset = align_to_page(manifest_offset + manifest_size, MLLM_PAGE_ALIGNMENT)
    weights_offset = current_offset

    descriptors = []
    tensor_payloads = []
    tensor_offsets = {}

    for t in tensors_to_write:
        name = t["name"]
        shared_name = t.get("shared_with")
        if shared_name and shared_name in tensor_offsets:
            tensor_offset = tensor_offsets[shared_name][0]
            data_len = tensor_offsets[shared_name][1]
            payload = None
        else:
            data_len = len(t["data"])
            tensor_offset = current_offset
            tensor_offsets[name] = (tensor_offset, data_len)
            payload = t["data"]
            current_offset = align_to_page(current_offset + data_len, MLLM_PAGE_ALIGNMENT)

        name_bytes = name.encode("utf-8")[:63].ljust(64, b'\x00')
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
        assert len(desc_bytes) == 128
        descriptors.append(desc_bytes)
        if payload is not None:
            tensor_payloads.append((tensor_offset, payload))

    total_file_size = current_offset

    header_bytes = struct.pack(
        "<IIIIIIIIIIIIQQQQ48s",
        MLLM_MAGIC,
        MLLM_VERSION,
        1, # Architecture: LLaMA
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
    assert len(header_bytes) == 128

    os.makedirs(os.path.dirname(output_mllm_path), exist_ok=True)
    with open(output_mllm_path, "wb") as f:
        f.write(header_bytes)
        for desc in descriptors:
            f.write(desc)

        for target_offset, payload in tensor_payloads:
            pos = f.tell()
            if pos < target_offset:
                f.write(b'\x00' * (target_offset - pos))
            f.write(payload)

        pos = f.tell()
        if pos < total_file_size:
            f.write(b'\x00' * (total_file_size - pos))

    file_mb = total_file_size / (1024 * 1024)
    elapsed = time.time() - t_start
    print(f"[Exporter] ✅ Generated '{output_mllm_path}' ({file_mb:.2f} MB, {num_tensors} tensors in {elapsed:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description="Export LLaMA 3.2 models to NanoEdge .mllm")
    parser.add_argument("--model", choices=["1b", "3b", "all"], default="all", help="Model to convert")
    args = parser.parse_args()

    home = os.path.expanduser("~")
    p_1b = os.path.join(home, ".cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/*")
    p_3b = os.path.join(home, ".cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/*")

    suite_dir = "models/suite"
    ios_res_dir = "mobile/ios/NanoEdgeApp/Resources"
    os.makedirs(suite_dir, exist_ok=True)
    os.makedirs(ios_res_dir, exist_ok=True)

    # 1. Export Shared LLaMA 3.2 Vocab (128,256 tokens)
    vocab_snapshots = glob.glob(p_1b)
    if vocab_snapshots:
        export_vocab(vocab_snapshots[0], os.path.join(suite_dir, "llama3_vocab.json"))
        export_vocab(vocab_snapshots[0], os.path.join(ios_res_dir, "llama3_vocab.json"))

    # 2. Export 1B Model
    if args.model in ["1b", "all"]:
        out_1b_suite = os.path.join(suite_dir, "llama3_2_1b_instruct_q4.mllm")
        out_1b_ios = os.path.join(ios_res_dir, "llama3_2_1b_instruct_q4.mllm")
        convert_model("LLaMA 3.2 1B Instruct", p_1b, out_1b_suite)
        if os.path.abspath(out_1b_suite) != os.path.abspath(out_1b_ios):
            import shutil
            print(f"[Exporter] Copying 1B model to {out_1b_ios}...")
            shutil.copyfile(out_1b_suite, out_1b_ios)

    # 3. Export 3B Model
    if args.model in ["3b", "all"]:
        out_3b_suite = os.path.join(suite_dir, "llama3_2_3b_instruct_q4.mllm")
        out_3b_ios = os.path.join(ios_res_dir, "llama3_2_3b_instruct_q4.mllm")
        convert_model("LLaMA 3.2 3B Instruct", p_3b, out_3b_suite)
        if os.path.abspath(out_3b_suite) != os.path.abspath(out_3b_ios):
            import shutil
            print(f"[Exporter] Copying 3B model to {out_3b_ios}...")
            shutil.copyfile(out_3b_suite, out_3b_ios)

    print("\n🎉 All LLaMA 3.2 models converted and staged successfully!")

if __name__ == "__main__":
    main()
