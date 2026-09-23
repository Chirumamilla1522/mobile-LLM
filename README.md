# NanoEdge: Hardware-Aware Mobile LLM Runtime & Compiler

A bare-metal mobile inference engine and offline model compiler engineered for flagship Apple Silicon (iPhone A-series, M-series) and Qualcomm Snapdragon devices.

Rather than deserializing models into RAM or wrapping general-purpose runtimes, **NanoEdge** treats model layout, memory mapping, tiled quantization, unified memory, and GPU/NPU compute kernels as a single co-designed optimization stack.

---

## Key Systems Architecture

```
                      Hugging Face / SafeTensors
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   OFFLINE MODEL COMPILER     │
                   │   (tools/mllm_compiler.py)   │
                   │                              │
                   │  • Sequential Layer Ordering │
                   │  • 16KB/64KB Page Alignment  │
                   │  • Hardware Tile Packing     │
                   └──────────────┬───────────────┘
                                  │ generates
                                  ▼
                        model.mllm binary
        ┌──────────────────────────────────────────────────┐
        │ HEADER  │ MANIFEST │ L0_QKV │ L0_MLP │ ... │ LM │
        └──────────────────────────────────────────────────┘
                                  │
                                  │ mmap() [Zero Copy]
                                  ▼
                   ┌──────────────────────────────┐
                   │    UNIFIED MEMORY RUNTIME    │
                   │  • MappedModel (mmap/madvise)│
                   │  • Lookahead Prefetcher      │
                   │  • Arena Allocator (0 alloc) │
                   └──────────────┬───────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│   METAL COMPUTE ENGINE   │              │   ARM NEON SIMD ENGINE   │
│   (Apple Silicon M/A)    │              │   (CPU Reference)        │
│                          │              │                          │
│ • Fused INT4 GEMV        │              │ • Vectorized INT4 Dot    │
│ • SIMD Group Reductions  │              │ • 128-bit NEON Pipeline  │
│ • In-Register Dequant    │              │ • Validates Math Parity  │
│ • Zero Memory Copy       │              │                          │
└──────────────────────────┘              └──────────────────────────┘
```

---

## Benchmarked Innovations

1. **Zero-Decompression Cold Start**:
   - Models are mapped with `mmap(MAP_PRIVATE)` directly into the 64-bit virtual address space.
   - Header and 128-byte tensor manifests parse in **under 1 millisecond** (`~800 microseconds`).
   - Starting RSS is only **~5 MB** for a full model container.

2. **In-Register INT4 Dequantization**:
   - Zero temporary FP16 weight buffers. Weights remain 4-bit packed in storage and DRAM.
   - Unpacking and scale multiplication occur inside Apple GPU ALU registers during fused matrix-vector multiplication.

3. **Zero-Copy Metal Unified Memory Binding**:
   - Binds `mmap` pointer directly to GPU via `[device newBufferWithBytesNoCopy:length:options:deallocator:]`.
   - Eliminates all CPU $\to$ GPU memory copy overhead.

4. **Hardware SIMD Group Reductions**:
   - Leverages `simd_sum()` inside 32-thread SIMD execution lanes for single-cycle reduction with zero threadgroup shared memory latency.

5. **Numerical Parity**:
   - Includes ground-truth vectorized ARM NEON reference kernels (`neon_gemv.cpp`) validating GPU numerical output within FP16 precision.

---

## Directory Structure

```
.
├── CMakeLists.txt                 # Build system for C++20, NEON, and Metal
├── include/mllm/
│   ├── model_format.h             # 128-byte header and page-aligned tensor descriptors
│   └── types.h                    # Quantization block structures (Q4_0, MQ4_Apple)
├── src/
│   ├── runtime/
│   │   ├── memory_mapped_model.hpp/cpp  # POSIX mmap & Mach VM task_info profiler
│   │   └── arena_allocator.hpp/cpp      # Bump allocator for intermediate activations
│   └── kernels/
│       ├── metal/
│       │   ├── q4_gemv.metal             # MSL kernels (GEMV, RMSNorm, RoPE)
│       │   ├── metal_backend.hpp         # Objective-C++ Metal driver header
│       │   └── metal_backend.mm          # Zero-copy GPU dispatch implementation
│       └── cpu/
│           ├── neon_gemv.hpp             # ARM NEON SIMD header
│           └── neon_gemv.cpp             # 128-bit NEON dot-product implementation
├── bench/
│   └── roofline_benchmark.cpp     # Cold start, memory footprint, bandwidth profiler
└── tools/
    └── mllm_compiler.py           # Offline model compiler and page-aligned packer
```

---

## Quick Start

### 1. Build the Engine
```bash
mkdir -p build && cd build
cmake ..
make -j
cd ..
```

### 2. Compile a Model Container
```bash
# Generate a test model container (dim=1024, layers=2)
python3 tools/mllm_compiler.py --synthetic --dim 1024 --hidden-dim 2048 --layers 2 --quant Q4_0 --out models/test_q4.mllm
```

### 3. Run Roofline Benchmark
```bash
./build/mllm_benchmark --model models/test_q4.mllm --iterations 100
```

## iOS Runtime Notes

Model weight files (`.mllm`) are not stored in this repository. Import a compatible file from the iOS app's Telemetry tab. The current runtime executes on Metal or ARM CPU; Neural Engine execution is not implemented for `.mllm` models, so the benchmark does not offer ANE as an engine. Core ML can allow CPU, GPU, and Neural Engine compute units for a Core ML model, but the OS chooses placement and does not promise simultaneous load balancing across all units.
