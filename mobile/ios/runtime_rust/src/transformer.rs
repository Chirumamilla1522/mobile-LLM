use crate::kernels::{
    apply_rope, fp16_to_fp32, fp32_to_fp16, gemv_q4_0, gemv_q4_0_swiglu, gemv_q8_0,
    rms_norm,
};
use crate::model::MemoryMappedModel;
use crate::metal::MetalGemv;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MAX_SEQ_LEN: usize = 2048;
const KV_PAGE_ELEMENTS: usize = 8192; // 16 KiB of FP16, matching Apple VM pages.

struct PagedFp16Cache {
    pages: Vec<Option<Box<[u16]>>>,
}

impl PagedFp16Cache {
    fn new(elements: usize) -> Self {
        Self { pages: (0..elements.div_ceil(KV_PAGE_ELEMENTS)).map(|_| None).collect() }
    }

    fn get(&self, index: usize) -> u16 {
        self.pages[index / KV_PAGE_ELEMENTS]
            .as_ref()
            .map_or(0, |page| page[index % KV_PAGE_ELEMENTS])
    }

    fn set(&mut self, index: usize, value: u16) {
        let page = &mut self.pages[index / KV_PAGE_ELEMENTS];
        let page = page.get_or_insert_with(|| vec![0; KV_PAGE_ELEMENTS].into_boxed_slice());
        page[index % KV_PAGE_ELEMENTS] = value;
    }
}

fn common_prefix_len(left: &[i32], right: &[i32]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

#[derive(Default, Clone)]
struct LayerWeights {
    in_norm_offset: usize,
    in_norm_len: usize,
    q_proj_offset: usize,
    q_proj_len: usize,
    k_proj_offset: usize,
    k_proj_len: usize,
    v_proj_offset: usize,
    v_proj_len: usize,
    o_proj_offset: usize,
    o_proj_len: usize,
    post_norm_offset: usize,
    post_norm_len: usize,
    gate_proj_offset: usize,
    gate_proj_len: usize,
    up_proj_offset: usize,
    up_proj_len: usize,
    down_proj_offset: usize,
    down_proj_len: usize,
}

pub struct TransformerEngine {
    pub model: Arc<MemoryMappedModel>,
    pub tokenizer: Tokenizer,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    context_capacity: usize,

    embed_offset: usize,
    _embed_len: usize,
    embed_quant_type: u32,
    final_norm_offset: usize,
    final_norm_len: usize,
    lm_head_offset: usize,
    lm_head_len: usize,
    lm_head_quant_type: u32,

    layers: Vec<LayerWeights>,

    // Pre-allocated static activation buffers (Zero heap allocation in decode loop)
    x: Vec<f32>,
    x_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    ffn: Vec<f32>,
    proj_buf: Vec<f32>,
    pub logits: Vec<f32>,
    sample_buf: Vec<f32>,
    metal: Option<MetalGemv>,
    metal_enabled: bool,
    kv_precision_bits: u32,
    last_forward_metal: bool,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,

    // FP16 KV cache: [layer, kv_head, position, head_dim]
    kv_cache_k: PagedFp16Cache,
    kv_cache_v: PagedFp16Cache,
    cached_prompt_tokens: Vec<i32>,

    pub cancel_flag: Arc<AtomicBool>,
}

impl TransformerEngine {
    pub fn new(model: Arc<MemoryMappedModel>, tokenizer: Tokenizer) -> Result<Self, String> {
        let hdr = &model.header;
        let num_layers = { if hdr.num_layers > 0 { hdr.num_layers as usize } else { 30 } };
        let hidden_dim = { if hdr.hidden_dim > 0 { hdr.hidden_dim as usize } else { 576 } };
        let intermediate_dim = { if hdr.intermediate_dim > 0 { hdr.intermediate_dim as usize } else { 1536 } };
        let num_heads = { if hdr.num_heads > 0 { hdr.num_heads as usize } else { 9 } };
        let num_kv_heads = { if hdr.num_kv_heads > 0 { hdr.num_kv_heads as usize } else { 3 } };
        let head_dim = hidden_dim / num_heads.max(1);
        let vocab_size = { if hdr.vocab_size > 0 { hdr.vocab_size as usize } else { 49152 } };
        let rope_theta = if vocab_size > 50000 { 500000.0f32 } else { 100000.0f32 };
        let context_capacity = if hdr.max_seq_len > 0 {
            (hdr.max_seq_len as usize).min(MAX_SEQ_LEN)
        } else {
            MAX_SEQ_LEN
        };

        let mut embed_offset = 0;
        let mut embed_len = 0;
        let mut embed_quant_type = 1; // default FP16
        let mut final_norm_offset = 0;
        let mut final_norm_len = 0;
        let mut lm_head_offset = 0;
        let mut lm_head_len = 0;
        let mut lm_head_quant_type = 3; // default Q4_0

        let mut layers = vec![LayerWeights::default(); num_layers];

        for desc in &model.descriptors {
            let name = desc.name_str();
            let off = { desc.offset as usize };
            let sz = { desc.size_bytes as usize };
            let layer_idx = { desc.layer_idx };
            let tensor_type = { desc.tensor_type };
            let quant_type = { desc.quant_type };

            if name == "model.embed_tokens.weight" {
                embed_offset = off;
                embed_len = sz;
                embed_quant_type = quant_type;
            } else if name == "model.norm.weight" {
                final_norm_offset = off;
                final_norm_len = sz;
            } else if name == "lm_head.weight" {
                lm_head_offset = off;
                lm_head_len = sz;
                lm_head_quant_type = quant_type;
            } else if layer_idx >= 0 && (layer_idx as usize) < num_layers {
                let l = layer_idx as usize;
                match tensor_type {
                    5 => { // ATTN_NORM
                        layers[l].in_norm_offset = off;
                        layers[l].in_norm_len = sz;
                    }
                    1 => { // ATTN_Q
                        layers[l].q_proj_offset = off;
                        layers[l].q_proj_len = sz;
                    }
                    2 => { // ATTN_K
                        layers[l].k_proj_offset = off;
                        layers[l].k_proj_len = sz;
                    }
                    3 => { // ATTN_V
                        layers[l].v_proj_offset = off;
                        layers[l].v_proj_len = sz;
                    }
                    4 => { // ATTN_OUT
                        layers[l].o_proj_offset = off;
                        layers[l].o_proj_len = sz;
                    }
                    9 => { // FFN_NORM
                        layers[l].post_norm_offset = off;
                        layers[l].post_norm_len = sz;
                    }
                    6 => { // FFN_GATE
                        layers[l].gate_proj_offset = off;
                        layers[l].gate_proj_len = sz;
                    }
                    7 => { // FFN_UP
                        layers[l].up_proj_offset = off;
                        layers[l].up_proj_len = sz;
                    }
                    8 => { // FFN_DOWN
                        layers[l].down_proj_offset = off;
                        layers[l].down_proj_len = sz;
                    }
                    _ => {}
                }
            }
        }

        if lm_head_offset == 0 && embed_offset != 0 {
            lm_head_offset = embed_offset;
            lm_head_len = embed_len;
            lm_head_quant_type = embed_quant_type;
        }

        let kv_size = num_layers * context_capacity * num_kv_heads * head_dim;
        let proj_max = hidden_dim.max(intermediate_dim);
        let half_head_dim = head_dim / 2;
        let mut rope_cos = vec![0.0f32; context_capacity * half_head_dim];
        let mut rope_sin = vec![0.0f32; context_capacity * half_head_dim];
        for pos in 0..context_capacity {
            for pair in 0..half_head_dim {
                let angle = (pos as f32)
                    / rope_theta.powf((2 * pair) as f32 / head_dim as f32);
                rope_cos[pos * half_head_dim + pair] = angle.cos();
                rope_sin[pos * half_head_dim + pair] = angle.sin();
            }
        }

        let (model_ptr, model_len) = model.mapped_bytes();
        let metal = MetalGemv::new(model_ptr, model_len, context_capacity, rope_theta);
        let kv_precision_bits = if metal.is_some() { 8 } else { 16 };

        Ok(Self {
            model,
            tokenizer,
            num_layers,
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rope_theta,
            context_capacity,

            embed_offset,
            _embed_len: embed_len,
            embed_quant_type,
            final_norm_offset,
            final_norm_len,
            lm_head_offset,
            lm_head_len,
            lm_head_quant_type,

            layers,

            x: vec![0.0f32; hidden_dim],
            x_norm: vec![0.0f32; hidden_dim],
            q: vec![0.0f32; hidden_dim],
            k: vec![0.0f32; num_kv_heads * head_dim],
            v: vec![0.0f32; num_kv_heads * head_dim],
            attn_out: vec![0.0f32; hidden_dim],
            ffn: vec![0.0f32; intermediate_dim],
            proj_buf: vec![0.0f32; proj_max],
            logits: vec![0.0f32; vocab_size],
            sample_buf: vec![0.0f32; vocab_size],
            metal,
            metal_enabled: true,
            kv_precision_bits,
            last_forward_metal: false,
            rope_cos,
            rope_sin,

            kv_cache_k: PagedFp16Cache::new(kv_size),
            kv_cache_v: PagedFp16Cache::new(kv_size),
            cached_prompt_tokens: Vec::new(),

            cancel_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn forward_token(&mut self, token_id: i32, pos: usize, compute_logits: bool) {
        self.last_forward_metal = self.metal_enabled && self.metal.as_mut().is_some_and(|metal| {
            metal.forward(token_id, pos, compute_logits, None)
        });
        if self.last_forward_metal {
            return;
        }

        // 1. Embedding lookup (Q4_0 or FP16 -> FP32)
        if self.embed_quant_type == 3 {
            let blocks_per_row = self.hidden_dim / 32;
            let row_stride = blocks_per_row * 18;
            let row_start = self.embed_offset + (token_id as usize) * row_stride;
            let row_bytes = self.model.get_slice_at(row_start, row_stride);
            if row_bytes.len() >= row_stride {
                for b in 0..blocks_per_row {
                    let b_start = b * 18;
                    let scale_bits = u16::from_le_bytes([row_bytes[b_start], row_bytes[b_start + 1]]);
                    let scale = fp16_to_fp32(scale_bits);
                    for j in 0..16 {
                        let byte_val = row_bytes[b_start + 2 + j];
                        let q0 = ((byte_val & 0x0F) as i32) - 8;
                        let q1 = ((byte_val >> 4) as i32) - 8;
                        self.x[b * 32 + j * 2]     = (q0 as f32) * scale;
                        self.x[b * 32 + j * 2 + 1] = (q1 as f32) * scale;
                    }
                }
            }
        } else {
            let emb_stride = self.hidden_dim * 2;
            let emb_start = self.embed_offset + (token_id as usize) * emb_stride;
            let emb_bytes = self.model.get_slice_at(emb_start, emb_stride);
            if emb_bytes.len() >= emb_stride {
                for i in 0..self.hidden_dim {
                    let bits = u16::from_le_bytes([emb_bytes[i * 2], emb_bytes[i * 2 + 1]]);
                    self.x[i] = fp16_to_fp32(bits);
                }
            }
        }

        let gqa_ratio = self.num_heads / self.num_kv_heads;
        let attn_scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let kv_dim = self.num_kv_heads * self.head_dim;

        // 2. Causal Transformer Layers
        for l in 0..self.num_layers {
            let layer = &self.layers[l];

            // Input RMSNorm
            let in_norm_bytes = self.model.get_slice_at(layer.in_norm_offset, layer.in_norm_len);
            rms_norm(&self.x, in_norm_bytes, &mut self.x_norm, self.hidden_dim, 1e-5);

            // Q, K, V Projections
            let q_proj_bytes = self.model.get_slice_at(layer.q_proj_offset, layer.q_proj_len);
            gemv_q4_0(q_proj_bytes, &self.x_norm, &mut self.q, self.hidden_dim, self.hidden_dim);

            let k_proj_bytes = self.model.get_slice_at(layer.k_proj_offset, layer.k_proj_len);
            gemv_q4_0(k_proj_bytes, &self.x_norm, &mut self.k, self.hidden_dim, kv_dim);

            let v_proj_bytes = self.model.get_slice_at(layer.v_proj_offset, layer.v_proj_len);
            gemv_q4_0(v_proj_bytes, &self.x_norm, &mut self.v, self.hidden_dim, kv_dim);

            // Apply RoPE
            apply_rope(
                &mut self.q,
                self.num_heads,
                self.head_dim,
                &self.rope_cos,
                &self.rope_sin,
                pos,
            );
            apply_rope(
                &mut self.k,
                self.num_kv_heads,
                self.head_dim,
                &self.rope_cos,
                &self.rope_sin,
                pos,
            );

            // Save K and V using the decode-friendly [layer, kv_head, position, dim] layout.
            for kv_h in 0..self.num_kv_heads {
                let cache_offset = ((l * self.num_kv_heads + kv_h) * self.context_capacity + pos) * self.head_dim;
                let source_offset = kv_h * self.head_dim;
                for d in 0..self.head_dim {
                    self.kv_cache_k.set(cache_offset + d, fp32_to_fp16(self.k[source_offset + d]));
                    self.kv_cache_v.set(cache_offset + d, fp32_to_fp16(self.v[source_offset + d]));
                }
            }

            // Grouped-Query Attention (GQA)
            for h in 0..self.num_heads {
                let kv_h = h / gqa_ratio;
                let q_h = &self.q[h * self.head_dim..(h + 1) * self.head_dim];

                let out_h = &mut self.attn_out[h * self.head_dim..(h + 1) * self.head_dim];
                out_h.fill(0.0);
                let mut max_score = f32::NEG_INFINITY;
                let mut normalizer = 0.0f32;

                for t in 0..=pos {
                    let step_kv_off = ((l * self.num_kv_heads + kv_h) * self.context_capacity + t) * self.head_dim;

                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_h[d] * fp16_to_fp32(self.kv_cache_k.get(step_kv_off + d));
                    }
                    let score = dot * attn_scale;
                    let next_max = max_score.max(score);
                    let previous_scale = (max_score - next_max).exp();
                    let value_scale = (score - next_max).exp();

                    for d in 0..self.head_dim {
                        let value = fp16_to_fp32(self.kv_cache_v.get(step_kv_off + d));
                        out_h[d] = out_h[d] * previous_scale + value * value_scale;
                    }

                    normalizer = normalizer * previous_scale + value_scale;
                    max_score = next_max;
                }

                if normalizer > 0.0 {
                    let inverse = 1.0 / normalizer;
                    for value in out_h.iter_mut() {
                        *value *= inverse;
                    }
                }
            }

            // Attention Output Projection
            let o_proj_bytes = self.model.get_slice_at(layer.o_proj_offset, layer.o_proj_len);
            gemv_q4_0(o_proj_bytes, &self.attn_out, &mut self.proj_buf, self.hidden_dim, self.hidden_dim);

            // Residual connection
            for i in 0..self.hidden_dim {
                self.x[i] += self.proj_buf[i];
            }

            // Post-Attention RMSNorm
            let post_norm_bytes = self.model.get_slice_at(layer.post_norm_offset, layer.post_norm_len);
            rms_norm(&self.x, post_norm_bytes, &mut self.x_norm, self.hidden_dim, 1e-5);

            // SwiGLU MLP
            let gate_bytes = self.model.get_slice_at(layer.gate_proj_offset, layer.gate_proj_len);
            let up_bytes = self.model.get_slice_at(layer.up_proj_offset, layer.up_proj_len);
            gemv_q4_0_swiglu(
                gate_bytes,
                up_bytes,
                &self.x_norm,
                &mut self.ffn,
                self.hidden_dim,
                self.intermediate_dim,
            );

            // Down Projection
            let down_bytes = self.model.get_slice_at(layer.down_proj_offset, layer.down_proj_len);
            gemv_q4_0(down_bytes, &self.ffn, &mut self.proj_buf, self.intermediate_dim, self.hidden_dim);

            // Residual connection
            for i in 0..self.hidden_dim {
                self.x[i] += self.proj_buf[i];
            }
        }

        if !compute_logits {
            return;
        }

        // 3. Final RMSNorm
        let final_norm_bytes = self.model.get_slice_at(self.final_norm_offset, self.final_norm_len);
        rms_norm(&self.x, final_norm_bytes, &mut self.x_norm, self.hidden_dim, 1e-5);

        // 4. LM Head Projection to vocabulary logits
        let lm_head_bytes = self.model.get_slice_at(self.lm_head_offset, self.lm_head_len);
        if self.lm_head_quant_type == 2 {
            gemv_q8_0(lm_head_bytes, &self.x_norm, &mut self.logits, self.hidden_dim, self.vocab_size);
        } else {
            gemv_q4_0(lm_head_bytes, &self.x_norm, &mut self.logits, self.hidden_dim, self.vocab_size);
        }
    }

    pub fn set_metal_enabled(&mut self, enabled: bool) {
        if self.metal_enabled != enabled {
            self.cached_prompt_tokens.clear();
        }
        self.metal_enabled = enabled;
    }

    pub fn set_kv_precision(&mut self, bits: u32) -> bool {
        if bits != 8 && bits != 16 { return false; }
        if bits == self.kv_precision_bits { return true; }
        let changed = self.metal.as_mut().is_some_and(|metal| metal.set_kv_precision(bits))
            || (!self.metal_enabled && bits == 16);
        if changed {
            self.kv_precision_bits = bits;
            self.cached_prompt_tokens.clear();
        }
        changed
    }

    fn sample_next(&mut self, sampler: &mut Sampler, recent_tokens: &[i32]) -> i32 {
        if self.last_forward_metal {
            if let Some(token) = self.metal.as_mut().and_then(|metal| {
                metal.sample(
                    recent_tokens,
                    sampler.config.temperature,
                    sampler.config.repetition_penalty,
                    sampler.config.min_p,
                )
            }) {
                return token;
            }
        }
        sampler.sample(&self.logits, recent_tokens, &mut self.sample_buf)
    }

    pub fn generate<F, C>(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        config: SamplerConfig,
        mut on_token: F,
        on_complete: C,
    ) where
        F: FnMut(&str, f64) -> bool,
        C: FnOnce(&str, f64, f64, f64),
    {
        self.cancel_flag.store(false, Ordering::SeqCst);
        let start_time = Instant::now();

        let prompt_tokens = self.tokenizer.encode_prompt(prompt, system_prompt);
        let reusable = common_prefix_len(&self.cached_prompt_tokens, &prompt_tokens)
            .min(prompt_tokens.len().saturating_sub(1));
        let mut pos = reusable;

        // Prefill phase: Metal batches Q4 prompts in groups of 32; other formats use decode kernels.
        let remaining = &prompt_tokens[reusable..];
        let batched = remaining.len() >= 2 && self.metal_enabled && self.metal.as_mut().is_some_and(|metal| {
            metal.prefill(remaining, reusable)
        });
        if batched {
            pos += remaining.len();
            self.last_forward_metal = true;
        } else {
            for (index, &tok) in prompt_tokens.iter().enumerate().skip(reusable) {
                if pos >= self.context_capacity - 1 || self.cancel_flag.load(Ordering::Relaxed) {
                    break;
                }
                self.forward_token(tok, pos, index + 1 == prompt_tokens.len());
                pos += 1;
            }
        }
        self.cached_prompt_tokens.clone_from(&prompt_tokens);

        let first_token_time = Instant::now();
        let ttft_ms = first_token_time.duration_since(start_time).as_secs_f64() * 1000.0;

        // Decode phase
        let mut sampler = Sampler::new(config);
        let mut recent_tokens = Vec::with_capacity(64);
        let mut full_response = String::with_capacity(512);
        let mut generated_count = 0;

        let mut current_tok = self.sample_next(&mut sampler, &recent_tokens);

        while generated_count < sampler.config.max_tokens && pos < self.context_capacity - 1 {
            if self.cancel_flag.load(Ordering::Relaxed) {
                break;
            }
            if current_tok == self.tokenizer.eos_token_id || current_tok == 0 || current_tok == 128001 || current_tok == 128009 {
                break;
            }

            recent_tokens.push(current_tok);
            if recent_tokens.len() > 64 {
                recent_tokens.remove(0);
            }

            let step_start = Instant::now();
            let next_from_fused_metal = if self.metal_enabled {
                self.metal.as_mut().and_then(|metal| metal.forward_sample(
                    current_tok,
                    pos,
                    &recent_tokens,
                    sampler.config.temperature,
                    sampler.config.repetition_penalty,
                    sampler.config.min_p,
                ))
            } else {
                None
            };
            if next_from_fused_metal.is_some() {
                self.last_forward_metal = true;
            } else {
                self.forward_token(current_tok, pos, true);
            }
            pos += 1;
            let step_duration = step_start.elapsed().as_secs_f64();
            let instant_tok_s = if step_duration > 0.0001 { 1.0 / step_duration } else { 50.0 };

            let token_str = self.tokenizer.decode_token(current_tok);
            full_response.push_str(&token_str);
            generated_count += 1;

            let continue_gen = on_token(&token_str, instant_tok_s);
            if !continue_gen {
                break;
            }

            current_tok = next_from_fused_metal
                .unwrap_or_else(|| self.sample_next(&mut sampler, &recent_tokens));
        }

        let total_time_s = start_time.elapsed().as_secs_f64();
        let avg_tok_s = if total_time_s > 0.0 && generated_count > 0 {
            (generated_count as f64) / total_time_s
        } else {
            0.0
        };

        on_complete(&full_response, total_time_s, avg_tok_s, ttft_ms);
    }
}
