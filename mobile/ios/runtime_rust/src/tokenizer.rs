use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, i32>,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
}

impl Tokenizer {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref()).map_err(|e| format!("Failed to open vocab file: {}", e))?;
        let mut content = String::new();
        file.read_to_string(&mut content).map_err(|e| format!("Failed to read vocab file: {}", e))?;

        let bos = Self::parse_int_field(&content, "\"bos_token_id\":").unwrap_or(1);
        let eos = Self::parse_int_field(&content, "\"eos_token_id\":").unwrap_or(2);

        let tokens_key = "\"tokens\":";
        let tokens_pos = content.find(tokens_key).ok_or_else(|| "Missing 'tokens' in vocab".to_string())?;
        let start_bracket = content[tokens_pos..].find('[').ok_or_else(|| "Missing '['".to_string())? + tokens_pos;
        let end_bracket = content.rfind(']').ok_or_else(|| "Missing ']'".to_string())?;

        let slice = &content[start_bracket + 1..end_bracket];
        let bytes = slice.as_bytes();

        let mut vocab = Vec::with_capacity(131072);
        let mut token_to_id = HashMap::with_capacity(131072);

        let mut pos = 0;
        while pos < bytes.len() {
            while pos < bytes.len() && bytes[pos] != b'"' {
                pos += 1;
            }
            if pos >= bytes.len() {
                break;
            }
            let q1 = pos;
            pos += 1;

            while pos < bytes.len() {
                if bytes[pos] == b'"' && bytes[pos - 1] != b'\\' {
                    break;
                }
                pos += 1;
            }
            if pos >= bytes.len() {
                break;
            }
            let q2 = pos;
            pos += 1;

            let raw_str = &slice[q1 + 1..q2];
            let token_str = Self::unescape_json(raw_str);
            let id = vocab.len() as i32;
            token_to_id.insert(token_str.clone(), id);
            vocab.push(token_str);
        }

        Ok(Self {
            vocab,
            token_to_id,
            bos_token_id: bos,
            eos_token_id: eos,
        })
    }

    fn parse_int_field(content: &str, field: &str) -> Option<i32> {
        if let Some(pos) = content.find(field) {
            let sub = &content[pos + field.len()..];
            let trimmed = sub.trim_start();
            let mut end = 0;
            let bytes = trimmed.as_bytes();
            while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'-') {
                end += 1;
            }
            if end > 0 {
                return trimmed[..end].parse::<i32>().ok();
            }
        }
        None
    }

    fn unescape_json(s: &str) -> String {
        let mut res = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next_c) = chars.next() {
                    match next_c {
                        '"' => res.push('"'),
                        '\\' => res.push('\\'),
                        '/' => res.push('/'),
                        'b' => res.push('\x08'),
                        'f' => res.push('\x0c'),
                        'n' => res.push('\n'),
                        'r' => res.push('\r'),
                        't' => res.push('\t'),
                        'u' => {
                            let hex: String = chars.by_ref().take(4).collect();
                            if let Ok(u) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(u) {
                                    res.push(ch);
                                }
                            }
                        }
                        other => {
                            res.push('\\');
                            res.push(other);
                        }
                    }
                }
            } else {
                res.push(c);
            }
        }
        res
    }

    pub fn decode_token(&self, id: i32) -> String {
        if id < 0 || (id as usize) >= self.vocab.len() {
            return String::new();
        }

        let raw = &self.vocab[id as usize];
        if (raw.starts_with("<|") && raw.ends_with("|>")) || raw == "<|endoftext|>" {
            return String::new();
        }

        if raw.starts_with("<0x") && raw.ends_with('>') && raw.len() == 6 {
            if let Ok(byte_val) = u8::from_str_radix(&raw[3..5], 16) {
                if let Ok(s) = String::from_utf8(vec![byte_val]) {
                    return s;
                }
            }
        }

        raw.replace('Ġ', " ").replace('Ċ', "\n")
    }

    pub fn encode_prompt(&self, user_prompt: &str, system_prompt: &str) -> Vec<i32> {
        let mut tokens = Vec::with_capacity(128);

        // Check if vocabulary uses LLaMA 3.2 header tokens
        if self.token_to_id.contains_key("<|start_header_id|>") {
            let bot = *self.token_to_id.get("<|begin_of_text|>").unwrap_or(&128000);
            let start_h = *self.token_to_id.get("<|start_header_id|>").unwrap_or(&128006);
            let end_h = *self.token_to_id.get("<|end_header_id|>").unwrap_or(&128007);
            let eot = *self.token_to_id.get("<|eot_id|>").unwrap_or(&128009);

            tokens.push(bot);

            if !system_prompt.trim().is_empty() {
                tokens.push(start_h);
                self.tokenize_words_into("system", &mut tokens);
                tokens.push(end_h);
                self.tokenize_words_into("\n\n", &mut tokens);
                self.tokenize_words_into(system_prompt, &mut tokens);
                tokens.push(eot);
            }

            tokens.push(start_h);
            self.tokenize_words_into("user", &mut tokens);
            tokens.push(end_h);
            self.tokenize_words_into("\n\n", &mut tokens);
            self.tokenize_words_into(user_prompt, &mut tokens);
            tokens.push(eot);

            tokens.push(start_h);
            self.tokenize_words_into("assistant", &mut tokens);
            tokens.push(end_h);
            self.tokenize_words_into("\n\n", &mut tokens);
        } else {
            // ChatML Format (SmolLM2 / Qwen)
            tokens.push(1); // <|im_start|>
            tokens.push(9690); // system
            tokens.push(198); // \n
            self.tokenize_words_into(system_prompt, &mut tokens);
            tokens.push(2); // <|im_end|>
            tokens.push(198); // \n

            tokens.push(1); // <|im_start|>
            tokens.push(4093); // user
            tokens.push(198); // \n
            self.tokenize_words_into(user_prompt, &mut tokens);
            tokens.push(2); // <|im_end|>
            tokens.push(198); // \n

            tokens.push(1); // <|im_start|>
            tokens.push(520); // ass
            tokens.push(9531); // istant
            tokens.push(198); // \n
        }

        tokens
    }

    fn tokenize_words_into(&self, text: &str, out: &mut Vec<i32>) {
        let mut i = 0;
        let mut is_start = true;

        while i < text.len() {
            let ch = text[i..].chars().next().unwrap();
            let ch_len = ch.len_utf8();

            if ch == ' ' {
                is_start = false;
                i += ch_len;
                continue;
            }
            if ch == '\n' {
                out.push(198);
                is_start = true;
                i += ch_len;
                continue;
            }

            let mut j = i;
            while j < text.len() {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch == ' ' || next_ch == '\n' {
                    break;
                }
                j += next_ch.len_utf8();
            }

            let raw_word_str = &text[i..j];
            let bpe_word = if is_start {
                raw_word_str.to_string()
            } else {
                format!("Ġ{}", raw_word_str)
            };
            is_start = false;
            i = j;

            // 1. Exact match with BPE prefix
            if let Some(&id) = self.token_to_id.get(&bpe_word) {
                out.push(id);
                continue;
            }

            // 2. Exact match without prefix
            if let Some(&id) = self.token_to_id.get(raw_word_str) {
                out.push(id);
                continue;
            }

            // 3. Sub-word longest match
            let target = bpe_word;
            let target_chars: Vec<char> = target.chars().collect();
            let mut sub_i = 0;

            while sub_i < target_chars.len() {
                let mut best_id = -1;
                let mut best_len = 0;
                let max_l = (target_chars.len() - sub_i).min(32);

                for len in (1..=max_l).rev() {
                    let slice: String = target_chars[sub_i..sub_i + len].iter().collect();
                    if let Some(&id) = self.token_to_id.get(&slice) {
                        best_id = id;
                        best_len = len;
                        break;
                    }
                }

                if best_id != -1 {
                    out.push(best_id);
                    sub_i += best_len;
                } else {
                    sub_i += 1;
                }
            }
        }
    }
}
