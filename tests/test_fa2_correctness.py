#!/usr/bin/env python3
"""Test FA2 kernel correctness against reference attention."""
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
    sz = n * 2
    return device.newBufferWithLength_options_(sz, Metal.MTLResourceStorageModeShared)

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

def reference_attention(Q, K, V, sm_scale, window_left, window_right):
    """Reference sliding-window attention in float32."""
    seq_len = Q.shape[0]
    scores = (Q.astype(np.float32) @ K.astype(np.float32).T) * sm_scale
    for i in range(seq_len):
        for j in range(seq_len):
            if j < i - window_left or j > i + window_right:
                scores[i, j] = -1e9
    mx = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - mx)
    attn = exp_s / exp_s.sum(axis=1, keepdims=True)
    return (attn @ V.astype(np.float32))

kernel_path = os.path.join(os.path.dirname(__file__), "moonshine_metal", "flash_attention_fwd_32x32x64.metal")
source = open(kernel_path).read()
pipe = compile_kernel(source, "flash_attention_fwd")

def test_fa2(seq_len, n_heads, D, window_left, window_right, label="", verbose=False):
    np.random.seed(42)
    kv_dim = n_heads * D
    stride_h = D
    stride_m = kv_dim
    sm_scale = 1.0 / math.sqrt(D)

    # Pad to multiple of 32 to avoid OOB reads
    padded = ((seq_len + 31) // 32) * 32
    Q_all = np.zeros((padded, kv_dim), dtype=np.float16)
    K_all = np.zeros((padded, kv_dim), dtype=np.float16)
    V_all = np.zeros((padded, kv_dim), dtype=np.float16)
    Q_all[:seq_len] = (np.random.randn(seq_len, kv_dim) * 0.5).astype(np.float16)
    K_all[:seq_len] = (np.random.randn(seq_len, kv_dim) * 0.5).astype(np.float16)
    V_all[:seq_len] = (np.random.randn(seq_len, kv_dim) * 0.5).astype(np.float16)

    Q_buf = make_f16_buffer(Q_all)
    K_buf = make_f16_buffer(K_all)
    V_buf = make_f16_buffer(V_all)
    O_buf = make_zero_f16_buffer(padded * kv_dim)

    grid = ((seq_len + 31) // 32, n_heads, 1)
    tg = (832, 1, 1)

    dispatch(pipe, grid, tg, [
        Q_buf, K_buf, V_buf, O_buf,
        scalar_buf(seq_len, 'i'),
        scalar_buf(stride_h, 'i'),
        scalar_buf(stride_m, 'i'),
        scalar_buf(sm_scale, 'f'),
        scalar_buf(window_left, 'i'),
        scalar_buf(window_right, 'i'),
    ])

    O_gpu = read_f16_buffer(O_buf, padded * kv_dim).reshape(padded, kv_dim)[:seq_len]

    max_err = 0
    for h in range(n_heads):
        s = h * D
        e = s + D
        O_ref = reference_attention(Q_all[:seq_len, s:e], K_all[:seq_len, s:e],
                                     V_all[:seq_len, s:e], sm_scale, window_left, window_right)
        O_h = O_gpu[:, s:e].astype(np.float32)
        err = np.abs(O_ref - O_h)
        h_max = err.max()
        h_mean = err.mean()
        if verbose or h_max > 0.05:
            worst = np.unravel_index(err.argmax(), err.shape)
            print(f"  head {h}: max_err={h_max:.4f} mean={h_mean:.6f} "
                  f"worst_pos={worst} ref={O_ref[worst]:.4f} gpu={O_h[worst]:.4f}")
        max_err = max(max_err, h_max)

    status = "PASS" if max_err < 0.05 else "FAIL"
    print(f"[{status}] {label}seq={seq_len} heads={n_heads} win=[{window_left},{window_right}]: max_err={max_err:.4f}")
    return max_err

print("=== FA2 Kernel Correctness Tests ===\n")

# Single head (baseline)
test_fa2(64, 1, 64, 16, 0, "1head ")

# Two heads - isolate multi-head bug
print("\n--- Multi-head diagnosis ---")
test_fa2(64, 2, 64, 16, 0, "2heads ", verbose=True)
test_fa2(64, 3, 64, 16, 0, "3heads ", verbose=True)
test_fa2(64, 10, 64, 16, 0, "10heads ", verbose=True)

# Test: run each head separately with n_heads=1 stride
print("\n--- Single-head-at-a-time (n_heads=1 stride) ---")
np.random.seed(42)
kv_dim_10 = 10 * 64
Q_full = (np.random.randn(64, kv_dim_10) * 0.5).astype(np.float16)
K_full = (np.random.randn(64, kv_dim_10) * 0.5).astype(np.float16)
V_full = (np.random.randn(64, kv_dim_10) * 0.5).astype(np.float16)
sm_scale = 1.0 / math.sqrt(64)
for h in range(10):
    s = h * 64
    e = s + 64
    Qh = Q_full[:, s:e].copy()
    Kh = K_full[:, s:e].copy()
    Vh = V_full[:, s:e].copy()
    Q_buf = make_f16_buffer(Qh)
    K_buf = make_f16_buffer(Kh)
    V_buf = make_f16_buffer(Vh)
    O_buf = make_zero_f16_buffer(64 * 64)
    dispatch(pipe, (2, 1, 1), (832, 1, 1), [
        Q_buf, K_buf, V_buf, O_buf,
        scalar_buf(64, 'i'), scalar_buf(64, 'i'), scalar_buf(64, 'i'),
        scalar_buf(sm_scale, 'f'), scalar_buf(16, 'i'), scalar_buf(0, 'i'),
    ])
    O_h = read_f16_buffer(O_buf, 64*64).reshape(64, 64).astype(np.float32)
    O_ref = reference_attention(Qh, Kh, Vh, sm_scale, 16, 0)
    err = np.abs(O_ref - O_h).max()
    print(f"  head {h} (isolated): max_err={err:.4f}")
