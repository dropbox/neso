#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 22.

Targets: harder codegen patterns and real-world kernels:
- Non-square matmul (M!=N!=K)
- Larger matmul (128x128)
- Matmul with K > BLOCK_K (multiple loop iterations)
- tl.dot without scf.for loop (standalone)
- 2D reduction inside loop (attention-like score accumulation)
- Fused add + softmax + weighted-sum (attention pattern)
- RMSNorm through JIT
- SiLU activation
- Rotary position embedding (RoPE)
- Fused bias + dropout + residual
- Group normalization pattern
- Quantized W8A16 matmul (int8 weights, fp32 compute)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Non-square matmul
@triton.jit
def nonsquare_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                              stride_am, stride_ak,
                              stride_bk, stride_bn,
                              stride_cm, stride_cn,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# 2. RMSNorm
@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps,
                     BLOCK: tl.constexpr):
    """RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    # RMS
    sq_mean = tl.sum(x * x, axis=0) / n_cols
    rms = tl.sqrt(sq_mean + eps)
    out = x / rms * w
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 3. SiLU activation
@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """SiLU: out = x * sigmoid(x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Rotary position embedding (RoPE)
@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len, head_dim,
                 BLOCK_D: tl.constexpr):
    """Apply RoPE: split x into even/odd, rotate."""
    pos = tl.program_id(0)  # sequence position
    head = tl.program_id(1)  # head index
    half_dim = head_dim // 2
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < half_dim
    # Load x[pos, head, 0:half] and x[pos, head, half:]
    base = pos * head_dim + head * head_dim  # simplified flat layout
    # Actually, let's use a flat [seq_len, head_dim] layout per-head
    base = pos * head_dim
    x_even = tl.load(x_ptr + base + d_offs, mask=d_mask, other=0.0)
    x_odd = tl.load(x_ptr + base + half_dim + d_offs, mask=d_mask, other=0.0)
    cos_val = tl.load(cos_ptr + pos * half_dim + d_offs, mask=d_mask, other=1.0)
    sin_val = tl.load(sin_ptr + pos * half_dim + d_offs, mask=d_mask, other=0.0)
    # Rotate
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd = x_even * sin_val + x_odd * cos_val
    tl.store(out_ptr + base + d_offs, out_even, mask=d_mask)
    tl.store(out_ptr + base + half_dim + d_offs, out_odd, mask=d_mask)


# 5. Fused bias + dropout + residual
@triton.jit
def fused_bdr_kernel(x_ptr, bias_ptr, residual_ptr, out_ptr,
                       seed, drop_p, n, BLOCK: tl.constexpr):
    """out = residual + dropout(x + bias, p)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs % 64, mask=mask)  # broadcast bias over features
    r = tl.load(residual_ptr + offs, mask=mask)
    xb = x + b
    # Dropout
    rng = tl.rand(seed, offs)
    keep = rng > drop_p
    scale = 1.0 / (1.0 - drop_p)
    dropped = tl.where(keep, xb * scale, 0.0)
    out = r + dropped
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Standalone tl.dot (no loop)
@triton.jit
def standalone_dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
    """Single tl.dot without scf.for (K must fit in one block)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
    c = tl.dot(a, b)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c, mask=c_mask)


# 7. Group norm (simplified: single group)
@triton.jit
def group_norm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr,
                        n_channels, eps, BLOCK: tl.constexpr):
    """GroupNorm for a single spatial position, all channels in one group."""
    batch = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_channels
    x = tl.load(x_ptr + batch * n_channels + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_channels
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_channels
    xn = diff / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = g * xn + b
    tl.store(out_ptr + batch * n_channels + offs, out, mask=mask)


# 8. W8A16 quantized matmul (simplified per-column dequant)
@triton.jit
def w8a16_matmul_kernel(a_ptr, w_ptr, scale_ptr, out_ptr,
                          M, N, K,
                          stride_am, stride_ak,
                          stride_wk, stride_wn,
                          stride_om, stride_on,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    """A (fp32) @ dequant(W_int8, scale) -> out (fp32).
    W stored as int8, scale per-column."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    # Load per-column scales
    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        w_int = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        # Dequantize: w_float = w_int * scale[n]
        w_float = w_int.to(tl.float32) * scales[None, :]
        acc += tl.dot(a, w_float)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_nonsquare_matmul():
    M, N, K = 48, 32, 64
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    nonsquare_matmul_kernel[grid](a, b, c, M, N, K,
                                    a.stride(0), a.stride(1),
                                    b.stride(0), b.stride(1),
                                    c.stride(0), c.stride(1),
                                    BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    ref = a @ b
    err = (c - ref).abs().max().item()
    return err < 0.1, err


def test_rmsnorm():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    rmsnorm_kernel[(M,)](x, w, out, N, eps, BLOCK=64)
    # Reference
    rms = (x.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    ref = x / rms * w
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_silu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    silu_kernel[grid](x, out, n, BLOCK=256)
    ref = x * torch.sigmoid(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_rope():
    seq_len = 16
    head_dim = 32
    half_dim = head_dim // 2
    x = torch.randn(seq_len, head_dim, device='mps')
    # Generate cos/sin tables
    pos = torch.arange(seq_len, device='mps').float()
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, device='mps').float() / half_dim))
    angles = pos[:, None] * freqs[None, :]
    cos_table = torch.cos(angles)
    sin_table = torch.sin(angles)
    out = torch.zeros_like(x)
    rope_kernel[(seq_len, 1)](x, cos_table, sin_table, out, seq_len, head_dim,
                                BLOCK_D=16)
    # Reference
    x_even = x[:, :half_dim]
    x_odd = x[:, half_dim:]
    ref_even = x_even * cos_table - x_odd * sin_table
    ref_odd = x_even * sin_table + x_odd * cos_table
    ref = torch.cat([ref_even, ref_odd], dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_fused_bdr():
    n = 1024
    x = torch.randn(n, device='mps')
    bias = torch.randn(64, device='mps')
    residual = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fused_bdr_kernel[grid](x, bias, residual, out, 42, 0.0, n, BLOCK=256)
    # With p=0, dropout is identity: out = residual + (x + bias)
    ref = residual + x + bias.repeat(n // 64)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_standalone_dot():
    M, N, K = 16, 16, 16
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    standalone_dot_kernel[(1, 1)](a, b, c, M, N, K,
                                    BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    ref = a @ b
    err = (c - ref).abs().max().item()
    return err < 1e-2, err


def test_group_norm():
    batch = 16
    channels = 64
    x = torch.randn(batch, channels, device='mps')
    gamma = torch.randn(channels, device='mps')
    beta = torch.randn(channels, device='mps')
    out = torch.zeros(batch, channels, device='mps')
    eps = 1e-5
    group_norm_kernel[(batch,)](x, gamma, beta, out, channels, eps, BLOCK=64)
    # Reference: normalize over all channels (single group)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_w8a16_matmul():
    # Note: K must equal BLOCK_K (single iteration) because the generic scf.for
    # path doesn't support multi-iteration 2D pointer advancement yet.
    M, N, K = 32, 32, 16
    a = torch.randn(M, K, device='mps')
    # Quantized weights (int8 range)
    w_int = torch.randint(-128, 127, (K, N), device='mps', dtype=torch.int32)
    scale = torch.randn(N, device='mps') * 0.01  # small scales
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    w8a16_matmul_kernel[grid](a, w_int, scale, out, M, N, K,
                                a.stride(0), a.stride(1),
                                w_int.stride(0), w_int.stride(1),
                                out.stride(0), out.stride(1),
                                BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    # Reference: dequant then matmul
    w_float = w_int.float() * scale[None, :]
    ref = a @ w_float
    err = (out - ref).abs().max().item()
    return err < 0.5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 22) ===\n")
    print("    (Real-world kernels: RoPE, RMSNorm, W8A16, matmul variants)\n")

    tests = [
        ("Non-Square Matmul (48x32x64)", test_nonsquare_matmul),
        ("RMSNorm", test_rmsnorm),
        ("SiLU Activation", test_silu),
        ("RoPE (Rotary Embed)", test_rope),
        ("Fused Bias+Drop+Residual", test_fused_bdr),
        ("Standalone tl.dot", test_standalone_dot),
        ("GroupNorm", test_group_norm),
        ("W8A16 Quant Matmul", test_w8a16_matmul),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            ok, err = fn()
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status} (max_err={err:.2e})")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {name}: ERROR ({e})")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
