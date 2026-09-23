#[cfg(target_os = "ios")]
mod platform {
    use std::ffi::c_void;

    extern "C" {
        fn nanoedge_metal_bridge_create(
            model: *const c_void,
            length: usize,
            context: u32,
            rope_theta: f32,
        ) -> *mut c_void;
        fn nanoedge_metal_bridge_destroy(handle: *mut c_void);
        fn nanoedge_metal_bridge_forward(
            handle: *mut c_void,
            token: u32,
            pos: u32,
            compute_logits: bool,
            logits: *mut f32,
        ) -> bool;
        fn nanoedge_metal_bridge_forward_sample(
            handle: *mut c_void,
            token: u32,
            pos: u32,
            recent: *const i32,
            count: u32,
            temperature: f32,
            penalty: f32,
            min_p: f32,
        ) -> i32;
        fn nanoedge_metal_bridge_sample(
            handle: *mut c_void,
            recent: *const i32,
            count: u32,
            temperature: f32,
            penalty: f32,
            min_p: f32,
        ) -> i32;
        fn nanoedge_metal_bridge_prefill(
            handle: *mut c_void,
            tokens: *const i32,
            count: u32,
            start_pos: u32,
        ) -> bool;
        fn nanoedge_metal_bridge_set_kv_precision(handle: *mut c_void, bits: u32) -> bool;
    }

    pub struct MetalGemv(*mut c_void);

    impl MetalGemv {
        pub fn new(model: *const u8, length: usize, context: usize, rope_theta: f32) -> Option<Self> {
            let handle = unsafe {
                nanoedge_metal_bridge_create(model.cast(), length, context as u32, rope_theta)
            };
            (!handle.is_null()).then_some(Self(handle))
        }

        pub fn forward(
            &mut self,
            token: i32,
            pos: usize,
            compute_logits: bool,
            logits: Option<&mut [f32]>,
        ) -> bool {
            let output = logits.map_or(std::ptr::null_mut(), |values| values.as_mut_ptr());
            unsafe {
                nanoedge_metal_bridge_forward(
                    self.0,
                    token as u32,
                    pos as u32,
                    compute_logits,
                    output,
                )
            }
        }

        pub fn sample(&mut self, recent: &[i32], temperature: f32, penalty: f32, min_p: f32) -> Option<i32> {
            let token = unsafe {
                nanoedge_metal_bridge_sample(
                    self.0,
                    recent.as_ptr(),
                    recent.len().min(64) as u32,
                    temperature,
                    penalty,
                    min_p,
                )
            };
            (token >= 0).then_some(token)
        }

        pub fn forward_sample(
            &mut self,
            token: i32,
            pos: usize,
            recent: &[i32],
            temperature: f32,
            penalty: f32,
            min_p: f32,
        ) -> Option<i32> {
            let token = unsafe {
                nanoedge_metal_bridge_forward_sample(
                    self.0,
                    token as u32,
                    pos as u32,
                    recent.as_ptr(),
                    recent.len().min(64) as u32,
                    temperature,
                    penalty,
                    min_p,
                )
            };
            (token >= 0).then_some(token)
        }

        pub fn prefill(&mut self, tokens: &[i32], start_pos: usize) -> bool {
            let mut position = start_pos;
            for chunk in tokens.chunks(32) {
                let ok = if chunk.len() == 1 {
                    self.forward(chunk[0], position, true, None)
                } else {
                    unsafe { nanoedge_metal_bridge_prefill(self.0, chunk.as_ptr(), chunk.len() as u32, position as u32) }
                };
                if !ok {
                    return false;
                }
                position += chunk.len();
            }
            true
        }

        pub fn set_kv_precision(&mut self, bits: u32) -> bool {
            unsafe { nanoedge_metal_bridge_set_kv_precision(self.0, bits) }
        }
    }

    impl Drop for MetalGemv {
        fn drop(&mut self) {
            unsafe { nanoedge_metal_bridge_destroy(self.0) }
        }
    }
}

#[cfg(not(target_os = "ios"))]
mod platform {
    pub struct MetalGemv;

    impl MetalGemv {
        pub fn new(_model: *const u8, _length: usize, _context: usize, _rope_theta: f32) -> Option<Self> {
            None
        }

        pub fn forward(
            &mut self,
            _token: i32,
            _pos: usize,
            _compute_logits: bool,
            _logits: Option<&mut [f32]>,
        ) -> bool {
            false
        }

        pub fn sample(&mut self, _recent: &[i32], _temperature: f32, _penalty: f32, _min_p: f32) -> Option<i32> {
            None
        }

        pub fn forward_sample(
            &mut self,
            _token: i32,
            _pos: usize,
            _recent: &[i32],
            _temperature: f32,
            _penalty: f32,
            _min_p: f32,
        ) -> Option<i32> {
            None
        }

        pub fn prefill(&mut self, _tokens: &[i32], _start_pos: usize) -> bool {
            false
        }

        pub fn set_kv_precision(&mut self, bits: u32) -> bool {
            bits == 16
        }
    }
}

pub use platform::MetalGemv;
