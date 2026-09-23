use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;

pub const MLLM_MAGIC: u32 = 0x4D4C4C4D; // "MLLM"
pub const MLLM_VERSION: u32 = 1;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ModelHeader {
    pub magic: u32,
    pub version: u32,
    pub architecture: u32,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub page_size: u32,
    pub num_tensors: u32,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub weights_offset: u64,
    pub total_file_size: u64,
    pub reserved: [u8; 48],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptor {
    pub name: [u8; 64],
    pub tensor_type: u32,
    pub layer_idx: i32,
    pub quant_type: u32,
    pub rows: u32,
    pub cols: u32,
    pub offset: u64,
    pub size_bytes: u64,
    pub scales_offset: u64,
    pub scales_bytes: u32,
    pub reserved: [u8; 16],
}

impl TensorDescriptor {
    pub fn name_str(&self) -> &str {
        let len = self.name.iter().position(|&c| c == 0).unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..len]).unwrap_or("")
    }
}

pub struct MemoryMappedModel {
    mmap_ptr: *mut u8,
    file_size: usize,
    pub header: ModelHeader,
    pub descriptors: Vec<TensorDescriptor>,
    name_to_desc: HashMap<String, usize>,
}

unsafe impl Send for MemoryMappedModel {}
unsafe impl Sync for MemoryMappedModel {}

impl MemoryMappedModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open model file: {}", e))?;
        let metadata = file.metadata().map_err(|e| format!("Failed to read metadata: {}", e))?;
        let file_size = metadata.len() as usize;

        if file_size < std::mem::size_of::<ModelHeader>() {
            return Err("Model file too small to contain header".to_string());
        }

        let fd = file.as_raw_fd();
        let mmap_res = unsafe {
            mmap(
                ptr::null_mut(),
                file_size,
                PROT_READ,
                MAP_PRIVATE,
                fd,
                0,
            )
        };

        if mmap_res == MAP_FAILED || mmap_res.is_null() {
            return Err("mmap failed on model file".to_string());
        }
        let mmap_ptr = mmap_res as *mut u8;

        let mmap_slice = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, file_size) };
        let header = unsafe { ptr::read_unaligned(mmap_slice.as_ptr() as *const ModelHeader) };

        let magic = { header.magic };
        if magic != MLLM_MAGIC {
            unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
            return Err(format!("Invalid magic 0x{:08X}, expected 0x{:08X}", magic, MLLM_MAGIC));
        }
        let version = header.version;
        if version != MLLM_VERSION {
            unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
            return Err(format!("Unsupported model version {version}"));
        }
        if header.total_file_size as usize != file_size {
            unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
            return Err("Header file size does not match mapped file".to_string());
        }

        let num_tensors = { header.num_tensors as usize };
        let desc_size = std::mem::size_of::<TensorDescriptor>();
        let manifest_off = { header.manifest_offset as usize };

        let Some(manifest_bytes) = num_tensors.checked_mul(desc_size) else {
            unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
            return Err("Manifest size overflow".to_string());
        };
        if manifest_off.checked_add(manifest_bytes).is_none_or(|end| end > file_size) {
            unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
            return Err("Manifest extends beyond file size".to_string());
        }

        let mut descriptors = Vec::with_capacity(num_tensors);
        let mut name_to_desc = HashMap::with_capacity(num_tensors);

        for i in 0..num_tensors {
            let offset = manifest_off + i * desc_size;
            let desc = unsafe { ptr::read_unaligned((mmap_slice.as_ptr().add(offset)) as *const TensorDescriptor) };
            let name = desc.name_str().to_string();
            let rows = desc.rows as usize;
            let cols = desc.cols as usize;
            let blocks = match desc.quant_type {
                0 => rows.checked_mul(cols).and_then(|n| n.checked_mul(4)),
                1 => rows.checked_mul(cols).and_then(|n| n.checked_mul(2)),
                2 if cols % 32 == 0 => rows.checked_mul(cols / 32).and_then(|n| n.checked_mul(34)),
                3 if cols % 32 == 0 => rows.checked_mul(cols / 32).and_then(|n| n.checked_mul(18)),
                4 if cols % 32 == 0 => rows.checked_mul(cols / 32).and_then(|n| n.checked_mul(20)),
                5 if cols % 256 == 0 => rows.checked_mul(cols / 256).and_then(|n| n.checked_mul(144)),
                _ => None,
            };
            if rows == 0 || cols == 0 || blocks != Some(desc.size_bytes as usize) {
                unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
                return Err(format!("Invalid dimensions or byte size for tensor {name}"));
            }
            let tensor_end = (desc.offset as usize).checked_add(desc.size_bytes as usize);
            let scales_end = (desc.scales_offset as usize).checked_add(desc.scales_bytes as usize);
            if tensor_end.is_none_or(|end| end > file_size)
                || (desc.scales_bytes > 0 && scales_end.is_none_or(|end| end > file_size))
            {
                unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
                return Err(format!("Tensor {name} extends beyond file size"));
            }
            if name_to_desc.contains_key(&name) {
                unsafe { munmap(mmap_ptr as *mut c_void, file_size) };
                return Err(format!("Duplicate tensor name {name}"));
            }
            name_to_desc.insert(name, i);
            descriptors.push(desc);
        }

        Ok(Self {
            mmap_ptr,
            file_size,
            header,
            descriptors,
            name_to_desc,
        })
    }

    pub fn get_tensor_data(&self, desc: &TensorDescriptor) -> &[u8] {
        let off = { desc.offset as usize };
        let sz = { desc.size_bytes as usize };
        self.get_slice_at(off, sz)
    }

    pub fn get_slice_at(&self, off: usize, sz: usize) -> &[u8] {
        if off.checked_add(sz).is_none_or(|end| end > self.file_size) {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.mmap_ptr.add(off), sz) }
        }
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.name_to_desc.get(name).map(|&idx| &self.descriptors[idx])
    }

    pub fn mapped_bytes(&self) -> (*const u8, usize) {
        (self.mmap_ptr.cast_const(), self.file_size)
    }
}

impl Drop for MemoryMappedModel {
    fn drop(&mut self) {
        if !self.mmap_ptr.is_null() && self.file_size > 0 {
            unsafe {
                munmap(self.mmap_ptr as *mut c_void, self.file_size);
            }
        }
    }
}
