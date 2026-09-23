pub struct SamplerConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.2,
            repetition_penalty: 1.15,
            min_p: 0.05,
        }
    }
}

// Simple, blazing fast Xorshift64 PRNG to avoid external rand dependency
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

pub struct Sampler {
    pub config: SamplerConfig,
    rng: FastRng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: FastRng::new(133742),
        }
    }

    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[i32], buffer: &mut [f32]) -> i32 {
        let v = logits.len();
        if v == 0 {
            return 2;
        }

        // Copy logits to pre-allocated working buffer (zero heap allocation!)
        let scaled = &mut buffer[..v];
        scaled.copy_from_slice(logits);

        // 1. Repetition penalty
        if self.config.repetition_penalty > 1.0 {
            for &tok in recent_tokens {
                if tok >= 0 && (tok as usize) < v {
                    let idx = tok as usize;
                    if scaled[idx] > 0.0 {
                        scaled[idx] /= self.config.repetition_penalty;
                    } else {
                        scaled[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature scaling
        let temp = self.config.temperature.max(0.01);
        for x in scaled.iter_mut() {
            *x /= temp;
        }

        // 3. Greedy sampling if temperature is very low
        if temp <= 0.05 {
            let mut best_idx = 0;
            let mut best_val = scaled[0];
            for (i, &val) in scaled.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            return best_idx as i32;
        }

        // 4. Min-P filtering
        let mut max_l = scaled[0];
        for &val in scaled.iter() {
            if val > max_l {
                max_l = val;
            }
        }

        let min_thresh = self.config.min_p; // exp(max_l - max_l) * min_p = 1.0 * min_p
        let mut sum_p = 0.0f32;

        for x in scaled.iter_mut() {
            let p = (*x - max_l).exp();
            if p >= min_thresh {
                *x = p;
                sum_p += p;
            } else {
                *x = 0.0;
            }
        }

        if sum_p <= 0.0 {
            let mut best_idx = 0;
            let mut best_val = logits[0];
            for (i, &val) in logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            return best_idx as i32;
        }

        let r = self.rng.next_f32() * sum_p;
        let mut accum = 0.0f32;

        for (i, &p) in scaled.iter().enumerate() {
            if p > 0.0 {
                accum += p;
                if accum >= r {
                    return i as i32;
                }
            }
        }

        2 // EOS fallback
    }
}
