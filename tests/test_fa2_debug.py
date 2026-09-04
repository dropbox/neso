#!/usr/bin/env python3
"""Debug FA2 kernel: compare per-row output for failing head."""
import sys, os, struct, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()

def make_f16_buffer(data_f16):
    raw = data_f16.tobytes()
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def make_zero_f16_buffer(n):
    return device.newBufferWithLength_options_(n * 2, Metal.MTLResourceStorageModeShared)

def scalar_buf(val, fmt='i'):
    raw = struct.pack(fmt, val)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def read_f16_buffer(buf, n):
    raw = buf.contents().as_buffer(n * 2)
    return np.frombuffer(raw, dtype=np.float16).copy()

def compile_kernel(source, name):
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    lib, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err:
        raise RuntimeError(f"Compile error: {err.localizedDescription()}")
    fn = lib.newFunctionWithName_(name)
    pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err:
        raise RuntimeError(f"Pipeline error: {err.localizedDescription()}")
    return pipe

def dispatch(pipe, grid, tg_size, buffers):
    cb = queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pipe)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*tg_size))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

def reference_attention_detail(Q, K, V, sm_scale, window_left, window_right):
    """Reference with detailed intermediates."""
    seq_len = Q.shape[0]
    scores = (Q.astype(np.float32) @ K.astype(np.float32).T) * sm_scale
    mask = np.full_like(scores, -1e9)
    for i in range(seq_len):
        for j in range(seq_len):
            if i - window_left <= j <= i + window_right:
                mask[i, j] = 0.0
    scores_masked = np.where(mask == 0, scores, -1e9)
    mx = scores_masked.max(axis=1, keepdims=True)
    exp_s = np.exp(scores_masked - mx)
    denom = exp_s.sum(axis=1, keepdims=True)
    attn = exp_s / denom
    return attn @ V.astype(np.float32), scores_masked, attn

kernel_path = os.path.join(os.path.dirname(__file__), "moonshine_metal", "flash_attention_fwd_32x32x64.metal")
source = open(kernel_path).read()
pipe = compile_kernel(source, "flash_attention_fwd")

# Reproduce failing head 5 from the test
np.random.seed(42)
seq_len = 64
D = 64
kv_dim_10 = 10 * D
Q_full = (np.random.randn(seq_len, kv_dim_10) * 0.5).astype(np.float16)
K_full = (np.random.randn(seq_len, kv_dim_10) * 0.5).astype(np.float16)
V_full = (np.random.randn(seq_len, kv_dim_10) * 0.5).astype(np.float16)

# Test head 5 isolated
h = 5
s, e = h * D, (h + 1) * D
Qh = Q_full[:, s:e].copy()
Kh = K_full[:, s:e].copy()
Vh = V_full[:, s:e].copy()

sm_scale = 1.0 / math.sqrt(D)
O_ref, scores_masked, attn_ref = reference_attention_detail(Qh, Kh, Vh, sm_scale, 16, 0)

# Run FA2 kernel
Q_buf = make_f16_buffer(Qh)
K_buf = make_f16_buffer(Kh)
V_buf = make_f16_buffer(Vh)
O_buf = make_zero_f16_buffer(seq_len * D)

dispatch(pipe, (2, 1, 1), (832, 1, 1), [
    Q_buf, K_buf, V_buf, O_buf,
    scalar_buf(64, 'i'), scalar_buf(64, 'i'), scalar_buf(64, 'i'),
    scalar_buf(sm_scale, 'f'), scalar_buf(16, 'i'), scalar_buf(0, 'i'),
])
O_gpu = read_f16_buffer(O_buf, seq_len * D).reshape(seq_len, D).astype(np.float32)

err = np.abs(O_ref - O_gpu)
print(f"Head 5 isolated: max_err={err.max():.4f}")
print(f"Per-row max error:")
for r in range(seq_len):
    row_err = err[r].max()
    if row_err > 0.01:
        worst_col = err[r].argmax()
        # Which KV block does this row's window span?
        win_start = max(0, r - 16)
        win_end = r  # window_right=0
        win_blocks = set()
        for j in range(win_start, win_end + 1):
            win_blocks.add(j // 32)
        print(f"  row {r}: max_err={row_err:.4f} col={worst_col} "
              f"ref={O_ref[r, worst_col]:.4f} gpu={O_gpu[r, worst_col]:.4f} "
              f"window=[{win_start},{win_end}] spans_blocks={sorted(win_blocks)}")

# Check: are errors concentrated on rows whose window spans two blocks?
print(f"\nRows spanning 2 KV blocks: 17-48 (window crosses block boundary)")
print(f"Rows in single block: 0-16 (block 0 only) and 49-63 (block 1 only)")
err_single = err[np.r_[0:17, 49:64]].max()
err_cross = err[17:49].max()
print(f"  Single-block rows max_err: {err_single:.4f}")
print(f"  Cross-block rows max_err:  {err_cross:.4f}")
