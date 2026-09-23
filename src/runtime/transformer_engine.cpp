#include "mllm/transformer_engine.hpp"
#include "kernels/cpu/neon_gemv.hpp"
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

namespace mllm {

TransformerEngine::TransformerEngine(std::shared_ptr<MemoryMappedModel> model)
    : model_(std::move(model)) {
    if (model_) {
        const auto& hdr = model_->header();
        num_layers_ = hdr.num_layers > 0 ? hdr.num_layers : 30;
        hidden_dim_ = hdr.hidden_dim > 0 ? hdr.hidden_dim : 576;
        intermediate_dim_ = hdr.intermediate_dim > 0 ? hdr.intermediate_dim : 1536;
        num_heads_ = hdr.num_heads > 0 ? hdr.num_heads : 9;
        num_kv_heads_ = hdr.num_kv_heads > 0 ? hdr.num_kv_heads : 3;
        head_dim_ = hidden_dim_ / std::max(num_heads_, 1u);

        // Pre-allocate static activation buffers to guarantee zero heap mallocs
        x_.resize(hidden_dim_, 0.0f);
        x_norm_.resize(hidden_dim_, 0.0f);
        q_.resize(hidden_dim_, 0.0f);
        k_.resize(num_kv_heads_ * head_dim_, 0.0f);
        v_.resize(num_kv_heads_ * head_dim_, 0.0f);
        attn_out_.resize(hidden_dim_, 0.0f);
        gate_.resize(intermediate_dim_, 0.0f);
        up_.resize(intermediate_dim_, 0.0f);
        ffn_.resize(intermediate_dim_, 0.0f);
        logits_.resize(hdr.vocab_size > 0 ? hdr.vocab_size : 49152, 0.0f);

        // Pre-allocate KV Cache
        size_t kv_size = num_layers_ * MAX_SEQ_LEN * num_kv_heads_ * head_dim_;
        kv_cache_k_.assign(kv_size, 0.0f);
        kv_cache_v_.assign(kv_size, 0.0f);

        bind_tensors();
    }
}

TransformerEngine::~TransformerEngine() = default;

void TransformerEngine::bind_tensors() {
    if (!model_) return;

    layers_.resize(num_layers_);

    for (const auto& desc : model_->descriptors()) {
        std::string name(desc.name);
        const void* ptr = model_->get_tensor_data(desc);

        if (name == "model.embed_tokens.weight") {
            embed_tokens_ = static_cast<const uint16_t*>(ptr);
        } else if (name == "model.norm.weight") {
            final_norm_ = static_cast<const uint16_t*>(ptr);
        } else if (name == "lm_head.weight") {
            lm_head_ = ptr;
            lm_head_quant_type_ = desc.quant_type;
        } else if (desc.layer_idx >= 0 && static_cast<uint32_t>(desc.layer_idx) < num_layers_) {
            uint32_t l = static_cast<uint32_t>(desc.layer_idx);
            auto type = static_cast<TensorType>(desc.tensor_type);

            switch (type) {
                case TensorType::ATTN_NORM:
                    layers_[l].in_norm = static_cast<const uint16_t*>(ptr);
                    break;
                case TensorType::ATTN_Q:
                    layers_[l].q_proj = ptr;
                    break;
                case TensorType::ATTN_K:
                    layers_[l].k_proj = ptr;
                    break;
                case TensorType::ATTN_V:
                    layers_[l].v_proj = ptr;
                    break;
                case TensorType::ATTN_OUT:
                    layers_[l].o_proj = ptr;
                    break;
                case TensorType::FFN_NORM:
                    layers_[l].post_norm = static_cast<const uint16_t*>(ptr);
                    break;
                case TensorType::FFN_GATE:
                    layers_[l].gate_proj = ptr;
                    break;
                case TensorType::FFN_UP:
                    layers_[l].up_proj = ptr;
                    break;
                case TensorType::FFN_DOWN:
                    layers_[l].down_proj = ptr;
                    break;
                default:
                    break;
            }
        }
    }
}

bool TransformerEngine::load_vocab(const std::string& vocab_json_path) {
    std::ifstream file(vocab_json_path);
    if (!file.is_open()) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    // Simple fast JSON parser for "tokens": ["...", "..."]
    size_t tokens_pos = content.find("\"tokens\":");
    if (tokens_pos == std::string::npos) return false;

    size_t start_bracket = content.find('[', tokens_pos);
    size_t end_bracket = content.rfind(']');
    if (start_bracket == std::string::npos || end_bracket == std::string::npos) return false;

    vocab_.clear();
    token_to_id_.clear();
    vocab_.reserve(49152);

    size_t pos = start_bracket + 1;
    while (pos < end_bracket) {
        size_t quote1 = content.find('"', pos);
        if (quote1 == std::string::npos || quote1 >= end_bracket) break;

        size_t quote2 = quote1 + 1;
        while (quote2 < end_bracket) {
            if (content[quote2] == '"' && content[quote2 - 1] != '\\') break;
            quote2++;
        }
        if (quote2 >= end_bracket) break;

        std::string tok = content.substr(quote1 + 1, quote2 - quote1 - 1);
        int32_t id = static_cast<int32_t>(vocab_.size());
        token_to_id_[tok] = id;
        vocab_.push_back(tok);

        pos = quote2 + 1;
    }

    vocab_loaded_ = !vocab_.empty();
    return vocab_loaded_;
}

std::string TransformerEngine::decode_token(int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_.size()) {
        return "";
    }
    std::string raw = vocab_[token_id];
    if (raw == "<|im_end|>" || raw == "<|endoftext|>" || raw == "<|im_start|>") {
        return "";
    }

    // Replace BPE marker 'Ġ' (0xC4 0xA0 in UTF-8) with space
    std::string result;
    result.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (static_cast<unsigned char>(raw[i]) == 0xC4 &&
            i + 1 < raw.size() &&
            static_cast<unsigned char>(raw[i+1]) == 0xA0) {
            result.push_back(' ');
            i++;
        } else if (static_cast<unsigned char>(raw[i]) == 0xC4 &&
                   i + 1 < raw.size() &&
                   static_cast<unsigned char>(raw[i+1]) == 0x8A) { // 'Ċ' -> '\n'
            result.push_back('\n');
            i++;
        } else {
            result.push_back(raw[i]);
        }
    }
    return result;
}

std::vector<int32_t> TransformerEngine::encode_prompt(const std::string& user_prompt, const std::string& system_prompt) {
    std::vector<int32_t> tokens;

    auto tokenize_words = [this, &tokens](const std::string& text) {
        size_t i = 0;
        bool is_start = true;

        while (i < text.size()) {
            // Skip leading whitespace or convert to space
            if (text[i] == ' ') {
                is_start = false;
                i++;
                continue;
            }
            if (text[i] == '\n') {
                tokens.push_back(198); // '\n' token
                is_start = true;
                i++;
                continue;
            }

            // Extract next word or punctuation sequence
            size_t j = i;
            while (j < text.size() && text[j] != ' ' && text[j] != '\n') {
                j++;
            }
            std::string raw_word = text.substr(i, j - i);
            std::string bpe_word = is_start ? raw_word : ("\xc4\xa0" + raw_word);
            is_start = false;
            i = j;

            // 1. Try matching the whole BPE word
            if (auto it = token_to_id_.find(bpe_word); it != token_to_id_.end()) {
                tokens.push_back(it->second);
                continue;
            }
            // 2. Try raw word without leading space
            if (auto it = token_to_id_.find(raw_word); it != token_to_id_.end()) {
                tokens.push_back(it->second);
                continue;
            }

            // 3. Sub-word longest match
            size_t sub_i = 0;
            std::string target = bpe_word;
            while (sub_i < target.size()) {
                int32_t best_id = -1;
                size_t best_len = 0;
                size_t max_l = std::min(target.size() - sub_i, (size_t)32);

                for (size_t len = max_l; len >= 1; --len) {
                    std::string slice = target.substr(sub_i, len);
                    if (auto it = token_to_id_.find(slice); it != token_to_id_.end()) {
                        best_id = it->second;
                        best_len = len;
                        break;
                    }
                }

                if (best_id != -1) {
                    tokens.push_back(best_id);
                    sub_i += best_len;
                } else {
                    sub_i++;
                }
            }
        }
    };

    // <|im_start|>system\n{system_prompt}<|im_end|>\n
    tokens.push_back(1); // <|im_start|>
    tokens.push_back(9690); // system
    tokens.push_back(198);  // \n
    tokenize_words(system_prompt);
    tokens.push_back(2);   // <|im_end|>
    tokens.push_back(198); // \n

    // <|im_start|>user\n{user_prompt}<|im_end|>\n
    tokens.push_back(1); // <|im_start|>
    tokens.push_back(4093); // user
    tokens.push_back(198);  // \n
    tokenize_words(user_prompt);
    tokens.push_back(2);   // <|im_end|>
    tokens.push_back(198); // \n

    // <|im_start|>assistant\n
    tokens.push_back(1);    // <|im_start|>
    tokens.push_back(520);  // ass
    tokens.push_back(9531); // istant
    tokens.push_back(198);  // \n

    return tokens;
}


void TransformerEngine::apply_rms_norm(const float* in, const uint16_t* weight, float* out, size_t dim) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += in[i] * in[i];
    }
    float mean_sq = sum_sq / static_cast<float>(dim);
    float inv_std = 1.0f / std::sqrt(mean_sq + rms_norm_eps_);

    for (size_t i = 0; i < dim; ++i) {
        float w = fp16_to_fp32(weight[i]);
        out[i] = in[i] * inv_std * w;
    }
}

void TransformerEngine::apply_rope(float* vec, size_t num_heads, size_t head_dim, int32_t pos) {
    size_t half_dim = head_dim / 2;
    for (size_t h = 0; h < num_heads; ++h) {
        float* head_ptr = vec + (h * head_dim);
        for (size_t i = 0; i < half_dim; ++i) {
            float theta = static_cast<float>(pos) / std::pow(rope_theta_, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float cos_th = std::cos(theta);
            float sin_th = std::sin(theta);

            float v0 = head_ptr[i];
            float v1 = head_ptr[i + half_dim];

            head_ptr[i]            = v0 * cos_th - v1 * sin_th;
            head_ptr[i + half_dim] = v0 * sin_th + v1 * cos_th;
        }
    }
}

void TransformerEngine::forward_token(int32_t token_id, int32_t pos) {
    if (!embed_tokens_ || !final_norm_ || !lm_head_) return;

    // 1. Embedding lookup (FP16 -> FP32)
    const uint16_t* emb_row = embed_tokens_ + (token_id * hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        x_[i] = fp16_to_fp32(emb_row[i]);
    }

    std::vector<float> proj_buf(std::max(hidden_dim_, intermediate_dim_), 0.0f);
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // 2. Transformer Layers
    for (uint32_t l = 0; l < num_layers_; ++l) {
        const auto& layer = layers_[l];
        if (!layer.in_norm || !layer.q_proj || !layer.k_proj || !layer.v_proj) continue;

        // Input RMSNorm
        apply_rms_norm(x_.data(), layer.in_norm, x_norm_.data(), hidden_dim_);

        // Q, K, V Matrix Multiplications (Q4_0 GEMV)
        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.q_proj),
            x_norm_.data(),
            q_.data(),
            hidden_dim_,
            hidden_dim_
        );

        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.k_proj),
            x_norm_.data(),
            k_.data(),
            hidden_dim_,
            num_kv_heads_ * head_dim_
        );

        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.v_proj),
            x_norm_.data(),
            v_.data(),
            hidden_dim_,
            num_kv_heads_ * head_dim_
        );

        // Apply RoPE
        apply_rope(q_.data(), num_heads_, head_dim_, pos);
        apply_rope(k_.data(), num_kv_heads_, head_dim_, pos);

        // Store K & V in KV Cache
        size_t kv_offset = (l * MAX_SEQ_LEN + pos) * (num_kv_heads_ * head_dim_);
        std::memcpy(kv_cache_k_.data() + kv_offset, k_.data(), num_kv_heads_ * head_dim_ * sizeof(float));
        std::memcpy(kv_cache_v_.data() + kv_offset, v_.data(), num_kv_heads_ * head_dim_ * sizeof(float));

        // Grouped-Query Attention (GQA 3:1)
        uint32_t gqa_ratio = num_heads_ / num_kv_heads_;

        for (uint32_t h = 0; h < num_heads_; ++h) {
            uint32_t kv_h = h / gqa_ratio;
            const float* q_h = q_.data() + (h * head_dim_);
            float* out_h = attn_out_.data() + (h * head_dim_);

            std::vector<float> scores(pos + 1, 0.0f);
            float max_score = -1e9f;

            for (int32_t t = 0; t <= pos; ++t) {
                size_t step_kv_off = (l * MAX_SEQ_LEN + t) * (num_kv_heads_ * head_dim_) + (kv_h * head_dim_);
                const float* k_t = kv_cache_k_.data() + step_kv_off;

                float dot = 0.0f;
                for (size_t d = 0; d < head_dim_; ++d) {
                    dot += q_h[d] * k_t[d];
                }
                scores[t] = dot * attn_scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            // Softmax with max-subtraction to prevent numerical overflow
            float sum_exp = 0.0f;
            for (int32_t t = 0; t <= pos; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

            // Attention weighted sum of V
            for (size_t d = 0; d < head_dim_; ++d) {
                float v_accum = 0.0f;
                for (int32_t t = 0; t <= pos; ++t) {
                    size_t step_kv_off = (l * MAX_SEQ_LEN + t) * (num_kv_heads_ * head_dim_) + (kv_h * head_dim_);
                    const float* v_t = kv_cache_v_.data() + step_kv_off;
                    v_accum += (scores[t] * inv_sum) * v_t[d];
                }
                out_h[d] = v_accum;
            }
        }

        // Out Projection GEMV
        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.o_proj),
            attn_out_.data(),
            proj_buf.data(),
            hidden_dim_,
            hidden_dim_
        );

        // Residual connection
        for (size_t i = 0; i < hidden_dim_; ++i) {
            x_[i] += proj_buf[i];
        }

        // Post-Attention RMSNorm
        apply_rms_norm(x_.data(), layer.post_norm, x_norm_.data(), hidden_dim_);

        // SwiGLU MLP (Gate, Up, Down)
        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.gate_proj),
            x_norm_.data(),
            gate_.data(),
            hidden_dim_,
            intermediate_dim_
        );

        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.up_proj),
            x_norm_.data(),
            up_.data(),
            hidden_dim_,
            intermediate_dim_
        );

        // SiLU(gate) * up
        for (size_t i = 0; i < intermediate_dim_; ++i) {
            float g = gate_[i];
            float silu = g / (1.0f + std::exp(-g));
            ffn_[i] = silu * up_[i];
        }

        // Down Projection GEMV
        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(layer.down_proj),
            ffn_.data(),
            proj_buf.data(),
            intermediate_dim_,
            hidden_dim_
        );

        // Residual connection
        for (size_t i = 0; i < hidden_dim_; ++i) {
            x_[i] += proj_buf[i];
        }
    }

    // 3. Final RMSNorm
    apply_rms_norm(x_.data(), final_norm_, x_norm_.data(), hidden_dim_);

    // 4. LM Head Projection to vocabulary logits
    if (lm_head_quant_type_ == static_cast<uint32_t>(QuantType::INT8) || lm_head_quant_type_ == 2) {
        NeonGemv::compute_q8_0(
            static_cast<const BlockQ8_0*>(lm_head_),
            x_norm_.data(),
            logits_.data(),
            hidden_dim_,
            static_cast<uint32_t>(logits_.size())
        );
    } else {
        NeonGemv::compute_q4_0(
            static_cast<const BlockQ4_0*>(lm_head_),
            x_norm_.data(),
            logits_.data(),
            hidden_dim_,
            static_cast<uint32_t>(logits_.size())
        );
    }
}

int32_t TransformerEngine::sample_next_token(const GenerationConfig& config, const std::vector<int32_t>& recent_tokens) {
    size_t V = logits_.size();
    std::vector<float> scaled_logits = logits_;

    // 1. Repetition penalty
    if (config.repetition_penalty > 1.0f) {
        for (int32_t tok : recent_tokens) {
            if (tok >= 0 && static_cast<size_t>(tok) < V) {
                if (scaled_logits[tok] > 0.0f) {
                    scaled_logits[tok] /= config.repetition_penalty;
                } else {
                    scaled_logits[tok] *= config.repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature scaling
    float temp = std::max(0.01f, config.temperature);
    for (size_t i = 0; i < V; ++i) {
        scaled_logits[i] /= temp;
    }

    // 3. Greedy sampling if temperature is very low
    if (temp <= 0.05f) {
        auto max_it = std::max_element(scaled_logits.begin(), scaled_logits.end());
        return static_cast<int32_t>(std::distance(scaled_logits.begin(), max_it));
    }

    // 4. Min-P filtering
    float max_l = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    float max_p = 1.0f; // exp(max_l - max_l)
    float min_thresh = config.min_p * max_p;

    std::vector<float> probs(V, 0.0f);
    float sum_p = 0.0f;

    for (size_t i = 0; i < V; ++i) {
        float p = std::exp(scaled_logits[i] - max_l);
        if (p >= min_thresh) {
            probs[i] = p;
            sum_p += p;
        }
    }

    if (sum_p <= 0.0f) {
        auto max_it = std::max_element(scaled_logits.begin(), scaled_logits.end());
        return static_cast<int32_t>(std::distance(scaled_logits.begin(), max_it));
    }

    // Random sample from distribution
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum_p);
    float r = dist(gen);

    float accum = 0.0f;
    for (size_t i = 0; i < V; ++i) {
        if (probs[i] > 0.0f) {
            accum += probs[i];
            if (accum >= r) {
                return static_cast<int32_t>(i);
            }
        }
    }

    return 2; // EOS fallback
}

void TransformerEngine::generate(
    const std::string& user_prompt,
    const std::string& system_prompt,
    const GenerationConfig& config,
    std::function<bool(const std::string& token, double tok_per_sec)> on_token,
    std::function<void(const std::string& full_text, double total_time_s, double avg_tok_s)> on_complete
) {
    if (!is_ready()) {
        if (on_complete) on_complete("Error: Model or vocabulary not loaded.", 0.0, 0.0);
        return;
    }

    std::vector<int32_t> prompt_tokens = encode_prompt(user_prompt, system_prompt);
    generate_from_tokens(prompt_tokens, config, on_token, on_complete);
}

void TransformerEngine::generate_from_tokens(
    const std::vector<int32_t>& prompt_tokens,
    const GenerationConfig& config,
    std::function<bool(const std::string& token, double tok_per_sec)> on_token,
    std::function<void(const std::string& full_text, double total_time_s, double avg_tok_s)> on_complete
) {
    if (!is_ready()) {
        if (on_complete) on_complete("Error: Model or vocabulary not loaded.", 0.0, 0.0);
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> tokens = prompt_tokens;
    if (tokens.empty()) {
        tokens.push_back(bos_token_id_);
    }

    // Prefill Phase: Process prompt tokens sequentially to populate KV cache
    int32_t pos = 0;
    for (int32_t tok : tokens) {
        if (pos >= static_cast<int32_t>(MAX_SEQ_LEN - 1)) break;
        forward_token(tok, pos++);
    }

    // Decode Phase: Autoregressive token generation
    std::vector<int32_t> recent_tokens;
    std::string full_response = "";
    int generated_count = 0;

    int32_t current_tok = sample_next_token(config, recent_tokens);

    while (generated_count < config.max_tokens && pos < static_cast<int32_t>(MAX_SEQ_LEN - 1)) {
        if (current_tok == eos_token_id_ || current_tok == 0) { // <|im_end|> or <|endoftext|>
            break;
        }

        auto step_start = std::chrono::high_resolution_clock::now();
        forward_token(current_tok, pos++);
        auto step_end = std::chrono::high_resolution_clock::now();

        double step_s = std::chrono::duration<double>(step_end - step_start).count();
        double instant_tok_s = step_s > 0.0001 ? (1.0 / step_s) : 60.0;

        std::string token_str = decode_token(current_tok);
        std::cout << "<ID:" << current_tok << ">";
        full_response += token_str;
        generated_count++;

        recent_tokens.push_back(current_tok);
        if (recent_tokens.size() > 64) {
            recent_tokens.erase(recent_tokens.begin());
        }

        if (on_token) {
            bool keep_going = on_token(token_str, instant_tok_s);
            if (!keep_going) break;
        }

        current_tok = sample_next_token(config, recent_tokens);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(end_time - start_time).count();
    double avg_tok_s = generated_count > 0 ? (static_cast<double>(generated_count) / total_time_s) : 0.0;

    if (on_complete) {
        on_complete(full_response, total_time_s, avg_tok_s);
    }
}

} // namespace mllm

