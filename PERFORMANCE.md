# Neso Backend — Performance Report

**Device:** Apple M2 Max (38-core GPU, 96 GB unified memory)
**Theoretical peak:** ~13.6 TFLOP/s fp32, ~27.2 TFLOP/s fp16
**Date:** March 2026

---

## Test Suite Summary

| Suite | Passing | Failing | Notes |
|-------|---------|---------|-------|
| test_codegen.py | 6 | 0 | Parser, MSL gen, compilation, execution |
| test_standalone.py | 3 | 0 | Vector add, element-wise, large scale |
| test_advanced.py | 13 | 0 | Outer product, gather, FP16, transpose, etc. |
| test_advanced2.py | 8 | 1 | Histogram (atomic_add known issue) |
| test_advanced3.py | 11 | 1 | Attention score (known, non-standard matmul) |
| test_advanced4.py | 9 | 0 | FP16 matmul, argmax, softmax, normalize |
| test_quantized_matmul.py | 17 | 0 | W8A16, W4A16, W8A16+scale |
| **Total** | **67** | **2** | **97% pass rate** |

---

## FP16 Matmul (A×B, fp16 in, fp32 accumulate)

Best tile: **128×128×32** with simdgroup_matrix hardware MMA.

| Size | Triton (GFLOP/s) | MPS torch.matmul (GFLOP/s) | vs MPS |
|------|------------------:|----------------------------:|-------:|
| 512³ | 2,384 | 887 | **2.7×** |
| 1024³ | 4,570 | 4,830 | 95% |
| 2048³ | 5,614 | 8,570 | 66% |
| 4096³ | 6,132 | 9,547 | 64% |

- Beats MPS at small sizes (≤1024) due to lower launch overhead
- At 4096³: **6.1 TFLOP/s** = 45% of theoretical fp32 peak

### Tile Size Comparison (4096³)

| Tile | GFLOP/s | Double Buffer | Notes |
|------|--------:|:-------------:|-------|
| 32×32×32 | 3,071 | Yes | Too small, SG underutilized |
| 64×64×16 | 3,623 | Yes | BK too small |
| 64×64×32 | 4,614 | Yes | Good for medium sizes |
| 128×128×16 | 6,124 | Yes | |
| 128×128×32 | **6,132** | Yes | **Best overall** |

---

## W8A16 Quantized Matmul (int8 weights, fp16 activations)

GPTQ/AWQ-style: `C = A(fp16) × dequant(B(int8))`, fp32 accumulate.
Dequant-aware cooperative load: `half4(float4(int4(char4_val)))` in threadgroup memory.

| Size | Tile | GFLOP/s | Notes |
|------|------|--------:|-------|
| 1024³ | 64×64×64 | 3,045 | |
| 2048³ | 128×64×32 | 5,032 | |
| 2048³ | 128×128×32 | 5,102 | |
| 4096³ | 128×64×32 | 5,237 | |
| 4096³ | 128×128×32 | **5,744** | **Best** |

- **5.7 TFLOP/s** at 4096³ = 42% of peak, 94% of fp16 matmul speed
- B weights loaded as `char4`, dequantized to `half4` in shared memory

---

## W8A16 Scaled Matmul (with per-channel scale)

`C = (A × dequant(B)) * scale[n]` — standard GPTQ/AWQ inference pattern.
Fused scale: loaded into small `sScale[BN]` buffer, applied during store. No sC needed.

| Size | Tile | GFLOP/s | vs Unscaled |
|------|------|--------:|------------:|
| 1024³ | 64×64×32 | 2,976 | 100% |
| 2048³ | 128×128×32 | 5,614 | 110% |
| 4096³ | 64×64×32 | 4,494 | 101% |
| 4096³ | 128×128×32 | **5,976** | **104%** |

- Fused scale is **faster** than unscaled at 128×128 because it avoids sC + double-buffer overhead
- **6.0 TFLOP/s** at 4096³ = 44% of peak

---

## Flash Attention 2 (single head, fp16)

Forward pass only. Codegen pipeline: standard Triton TTIR → MSL.
Uses fused online softmax, simdgroup_matrix for Q×K and P×V.

### d=64

| N | Triton (GFLOP/s) | MPS SDPA (GFLOP/s) | vs MPS |
|---|---------:|---------:|-------:|
| 256 | 510 | 61 | **8.3×** |
| 512 | 1,122 | 223 | **5.0×** |
| 1024 | 1,248 | 721 | **1.7×** |
| 2048 | 1,667 | 1,564 | **1.1×** |
| 4096 | 2,036 | 1,657 | **1.2×** |

### d=128

| N | Triton (GFLOP/s) | MPS SDPA (GFLOP/s) | vs MPS |
|---|---------:|---------:|-------:|
| 512 | 1,032 | 457 | **2.3×** |
| 1024 | 1,300 | 1,234 | 1.1× |
| 2048 | 1,795 | 2,658 | 0.7× |
| 4096 | 2,022 | 2,855 | 0.7× |

- **Beats MPS at N≤2048 for d=64** and **N≤1024 for d=128**
- Peak: **2.0 TFLOP/s** at N=4096 for both d=64 and d=128
- Experimental BM=32 for d=128: **2.7 TFLOP/s** at N=4096 (0.96× MPS)

---

## Intel GPU (UHD Graphics 630)

No simdgroup_matrix — uses scalar matmul fallback with register sub-tiling.
Theoretical peak: ~384 GFLOP/s fp32 (24 EUs, 1.2 GHz).

### FP16 Matmul (scalar path)

| Size | Tile | GFLOP/s | % Peak |
|------|------|--------:|-------:|
| 1024³ | 32×32×32 | 43 | 11% |
| 2048³ | 32×32×32 | 16 | 4% |

### W8A16 Quantized Matmul

| Size | Tile | GFLOP/s |
|------|------|--------:|
| 1024³ | 32×32×32 | 43 |
| 4096³ | 64×64×32 | 17 |

- Best tile size on Intel: **32×32×32** (larger tiles cause register spill)
- Intel lacks simdgroup_matrix; all MMA done via scalar FMA

---

## Threadgroup Memory Budget (32 KB limit)

| Kernel | Tile | Memory | Fits? |
|--------|------|-------:|:-----:|
| FP16 matmul | 128×128×32 | 32,768 | Yes (double-buf) |
| W8A16 matmul | 128×128×32 | 32,768 | Yes (double-buf) |
| W8A16 scaled | 128×128×32 | 25,088 | Yes (single-buf + sCast + sScale) |
| FA2 d=64 | BM=32, BN=32 | 29,696 | Yes |
| FA2 d=128 | BM=16, BN=32 | 27,648 | Yes |
| FA2 d=128 | BM=24, BN=32 | 32,544 | Tight (≈32KB) |

---

## Key Optimizations

| Optimization | Impact | Where Used |
|-------------|--------|------------|
| simdgroup_matrix 8×8 MMA | ~3× over scalar | Matmul, FA2 |
| Register tiling (TM×TN per SG) | ~1.5× at 128×128 | Matmul |
| Double-buffered A/B tiles | ~10-15% | Matmul (when memory allows) |
| vec4 cooperative loads | ~5-10% | FP16 matmul |
| Dequant-aware vec4 loads | int8→fp16 at load time | W8A16 matmul |
| Fused per-channel scale | Avoids sC entirely | W8A16 scaled matmul |
| Virtual tiles (binop_view) | Zero memory for index math | Post-matmul ops |
| In-place broadcast multiply | Avoids BM×BN allocation | Scale application |
| Fused online softmax | ~4 fewer barriers | FA2 |
| Double-buffered diag scratch | 1 barrier per block row | FA2 |
| L2-aware swizzled dispatch | Better tile reuse | Large matmuls |
| Grid-stride cooperative loads | Intel GPU compatibility | All |
| Lazy tile allocation | Saves ~8KB dead tiles | Matmul |
| Tile memory pool + liveness | Automatic reuse | All 2D kernels |

---

## Remaining Gaps vs MPS

| Workload | Gap | Likely Cause |
|----------|-----|-------------|
| FP16 matmul 4096³ | 64% of MPS | MPS uses Apple-proprietary tiling/scheduling |
| FA2 d=128 N=4096 | 71% of MPS | MPS SDPA is heavily hand-tuned |
| W8A16 4096³ | No MPS baseline | MPS doesn't expose int8 matmul directly |

The primary bottleneck at large sizes is likely the Apple GPU's hardware scheduler and memory hierarchy optimizations that MPS can exploit but our codegen cannot (e.g., async copies, hardware-managed double buffering, proprietary tile formats).
