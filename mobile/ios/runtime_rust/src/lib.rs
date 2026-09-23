pub mod kernels;
pub mod metal;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod transformer;

use model::MemoryMappedModel;
use sampler::SamplerConfig;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokenizer::Tokenizer;
use transformer::TransformerEngine;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NanoEdgeGenConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
}

pub type NanoEdgeTokenCallback = unsafe extern "C" fn(
    token: *const c_char,
    tok_per_sec: f64,
    user_data: *mut c_void,
) -> bool;

pub type NanoEdgeCompleteCallback = unsafe extern "C" fn(
    full_text: *const c_char,
    total_time_sec: f64,
    avg_tok_per_sec: f64,
    ttft_ms: f64,
    user_data: *mut c_void,
);

pub struct EngineHandle {
    pub engine: TransformerEngine,
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_init(
    model_path: *const c_char,
    vocab_path: *const c_char,
) -> *mut c_void {
    if model_path.is_null() || vocab_path.is_null() {
        return std::ptr::null_mut();
    }

    let model_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let vocab_str = match CStr::from_ptr(vocab_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let model = match MemoryMappedModel::load(model_str) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("[nanoedge_rust] Failed to load model: {}", e);
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::load(vocab_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[nanoedge_rust] Failed to load vocab: {}", e);
            return std::ptr::null_mut();
        }
    };

    let engine = match TransformerEngine::new(model, tokenizer) {
        Ok(eng) => eng,
        Err(e) => {
            eprintln!("[nanoedge_rust] Failed to initialize engine: {}", e);
            return std::ptr::null_mut();
        }
    };

    let handle = Box::new(EngineHandle { engine });
    Box::into_raw(handle) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_generate(
    handle: *mut c_void,
    prompt: *const c_char,
    system_prompt: *const c_char,
    config: *const NanoEdgeGenConfig,
    token_cb: Option<NanoEdgeTokenCallback>,
    complete_cb: Option<NanoEdgeCompleteCallback>,
    user_data: *mut c_void,
) -> bool {
    if handle.is_null() || prompt.is_null() {
        return false;
    }

    let engine_handle = &mut *(handle as *mut EngineHandle);
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let system_str = if !system_prompt.is_null() {
        CStr::from_ptr(system_prompt).to_str().unwrap_or("")
    } else {
        "You are a helpful, concise AI assistant running on Apple Silicon."
    };

    let sampler_cfg = if !config.is_null() {
        let c = &*config;
        SamplerConfig {
            max_tokens: if c.max_tokens > 0 { c.max_tokens as usize } else { 512 },
            temperature: c.temperature,
            repetition_penalty: c.repetition_penalty,
            min_p: c.min_p,
        }
    } else {
        SamplerConfig::default()
    };

    engine_handle.engine.generate(
        prompt_str,
        system_str,
        sampler_cfg,
        |token, tok_s| {
            if let Some(cb) = token_cb {
                if let Ok(c_tok) = CString::new(token) {
                    return cb(c_tok.as_ptr(), tok_s, user_data);
                }
            }
            true
        },
        |full_text, total_time, avg_tok, ttft| {
            if let Some(cb) = complete_cb {
                if let Ok(c_full) = CString::new(full_text) {
                    cb(c_full.as_ptr(), total_time, avg_tok, ttft, user_data);
                }
            }
        },
    );

    true
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_cancel(handle: *mut c_void) {
    if !handle.is_null() {
        let engine_handle = &mut *(handle as *mut EngineHandle);
        engine_handle.engine.cancel_flag.store(true, Ordering::SeqCst);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_set_execution_engine(handle: *mut c_void, engine: u32) {
    if !handle.is_null() {
        let engine_handle = &mut *(handle as *mut EngineHandle);
        engine_handle.engine.set_metal_enabled(engine <= 1);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_set_kv_precision(handle: *mut c_void, bits: u32) -> bool {
    if handle.is_null() { return false; }
    let engine_handle = &mut *(handle as *mut EngineHandle);
    engine_handle.engine.set_kv_precision(bits)
}

#[no_mangle]
pub unsafe extern "C" fn nanoedge_rust_free(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut EngineHandle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<model::ModelHeader>(), 128);
        assert_eq!(std::mem::size_of::<model::TensorDescriptor>(), 128);
    }

    #[test]
    fn test_fp16_conversion() {
        use kernels::{fp16_to_fp32, fp32_to_fp16};
        // 0.0 in FP16 = 0x0000
        assert_eq!(fp16_to_fp32(0x0000), 0.0);
        // 1.0 in FP16 = 0x3C00
        assert_eq!(fp16_to_fp32(0x3C00), 1.0);
        // -1.0 in FP16 = 0xBC00
        assert_eq!(fp16_to_fp32(0xBC00), -1.0);
        assert_eq!(fp32_to_fp16(1.0), 0x3C00);
        assert_eq!(fp32_to_fp16(-1.0), 0xBC00);
    }

    #[test]
    fn test_paged_prefix_cache_inference() {
        // The existing inference test covers cache writes; repeat generation exercises prefix reuse.
        let model_path = "../../../models/suite/smollm2_135m_q4.mllm";
        let vocab_path = "../../../models/suite/smollm2_vocab.json";
        if !std::path::Path::new(model_path).exists() { return; }
        let model = Arc::new(MemoryMappedModel::load(model_path).unwrap());
        let tokenizer = Tokenizer::load(vocab_path).unwrap();
        let mut engine = TransformerEngine::new(model, tokenizer).unwrap();
        engine.generate("hello", "", SamplerConfig { max_tokens: 1, ..Default::default() }, |_, _| true, |_, _, _, _| {});
        engine.generate("hello world", "", SamplerConfig { max_tokens: 1, ..Default::default() }, |_, _| true, |_, _, _, _| {});
    }

    #[test]
    fn test_rejects_overflowing_manifest() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&model::MLLM_MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&model::MLLM_VERSION.to_le_bytes());
        bytes[44..48].copy_from_slice(&1u32.to_le_bytes());
        bytes[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
        bytes[56..64].copy_from_slice(&128u64.to_le_bytes());
        bytes[72..80].copy_from_slice(&128u64.to_le_bytes());
        let path = std::env::temp_dir().join(format!("nanoedge-invalid-{}.mllm", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        assert!(MemoryMappedModel::load(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_smollm2_inference() {
        let model_path = CString::new("../../../models/suite/smollm2_135m_q4.mllm").unwrap();
        let vocab_path = CString::new("../../../models/suite/smollm2_vocab.json").unwrap();

        if !std::path::Path::new("../../../models/suite/smollm2_135m_q4.mllm").exists() {
            println!("Skipping end-to-end test: model file not found");
            return;
        }

        unsafe {
            let handle = nanoedge_rust_init(model_path.as_ptr(), vocab_path.as_ptr());
            assert!(!handle.is_null(), "nanoedge_rust_init returned null");

            let prompt = CString::new("The capital of France is").unwrap();
            let config = NanoEdgeGenConfig {
                max_tokens: 15,
                temperature: 0.1,
                repetition_penalty: 1.15,
                min_p: 0.05,
            };

            let mut generated = String::new();
            let ok = nanoedge_rust_generate(
                handle,
                prompt.as_ptr(),
                std::ptr::null(),
                &config,
                Some(test_token_cb),
                Some(test_complete_cb),
                &mut generated as *mut String as *mut c_void,
            );

            assert!(ok, "nanoedge_rust_generate failed");
            println!("\n=== RUST CORE GENERATED TOKENS ===");
            println!("{}", generated);
            println!("==================================\n");

            nanoedge_rust_free(handle);
        }
    }

    unsafe extern "C" fn test_token_cb(token: *const c_char, _tok_s: f64, user_data: *mut c_void) -> bool {
        let out = &mut *(user_data as *mut String);
        if let Ok(s) = CStr::from_ptr(token).to_str() {
            out.push_str(s);
        }
        true
    }

    unsafe extern "C" fn test_complete_cb(full_text: *const c_char, total_time: f64, avg_tok: f64, ttft: f64, _user_data: *mut c_void) {
        if let Ok(s) = CStr::from_ptr(full_text).to_str() {
            println!("[Rust Engine Complete] TTFT: {:.1} ms | Speed: {:.1} tok/s | Total Time: {:.3} s | Output: '{}'", ttft, avg_tok, total_time, s);
        }
    }

    #[test]
    fn test_llama3_2_1b_loading() {
        use crate::model::MemoryMappedModel;
        use crate::tokenizer::Tokenizer;

        let model_path = "../../../models/suite/llama3_2_1b_instruct_q4.mllm";
        let vocab_path = "../../../models/suite/llama3_vocab.json";

        if !std::path::Path::new(model_path).exists() {
            println!("Skipping LLaMA 1B test: model file not found");
            return;
        }

        let mmap_model = std::sync::Arc::new(MemoryMappedModel::load(model_path).expect("Failed loading LLaMA 1B"));
        let num_layers = { mmap_model.header.num_layers };
        let hidden_dim = { mmap_model.header.hidden_dim };
        let vocab_size = { mmap_model.header.vocab_size };
        assert_eq!(num_layers, 16);
        assert_eq!(hidden_dim, 2048);
        assert_eq!(vocab_size, 128256);
        println!("LLaMA 3.2 1B model loaded successfully: 16 layers, 2048 hidden, 128256 vocab");

        let tok = Tokenizer::load(vocab_path).expect("Failed loading LLaMA vocab");
        assert_eq!(tok.vocab.len(), 128256);
        let prompt_tokens = tok.encode_prompt("What is 2+2?", "Be concise");
        assert!(!prompt_tokens.is_empty());
        println!("LLaMA 3.2 prompt encoded into {} tokens: {:?}", prompt_tokens.len(), prompt_tokens);
    }

    #[test]
    fn test_llama3_2_3b_loading() {
        use crate::model::MemoryMappedModel;

        let model_path = "../../../models/suite/llama3_2_3b_instruct_q4.mllm";
        if !std::path::Path::new(model_path).exists() {
            println!("Skipping LLaMA 3B test: model file not found");
            return;
        }

        let mmap_model = std::sync::Arc::new(MemoryMappedModel::load(model_path).expect("Failed loading LLaMA 3B"));
        let num_layers = { mmap_model.header.num_layers };
        let hidden_dim = { mmap_model.header.hidden_dim };
        let vocab_size = { mmap_model.header.vocab_size };
        assert_eq!(num_layers, 28);
        assert_eq!(hidden_dim, 3072);
        assert_eq!(vocab_size, 128256);
        println!("LLaMA 3.2 3B model loaded successfully: 28 layers, 3072 hidden, 128256 vocab");
    }
}
