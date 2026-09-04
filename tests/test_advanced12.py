#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 12.

Targets: quantized matmul patterns (int8 weights with fp32/fp16 activations),
W4A16 dequantization, grouped quantization, and fused dequant+matmul.
These are the core building blocks for efficient LLM inference on Apple Silicon.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. W8A16 dequant matmul: fp16 activations * int8 weights (per-tensor scale)
@triton.jit
def w8a16_matmul_kernel(
    a_ptr, b_int8_ptr, scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Matmul with int8 weights: C = A @ (B_int8 * scale)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    scale = tl.load(scale_ptr)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_int8_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs).to(tl.float32)
        b_raw = tl.load(b_ptrs).to(tl.float32)
        b = b_raw * scale
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


# 2. Per-channel dequantization (each output channel has its own scale+zero)
@triton.jit
def per_channel_dequant_kernel(x_int8_ptr, scale_ptr, zero_ptr, out_ptr,
                                n_rows, n_cols,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Dequantize int8 with per-column scale and zero point."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < n_rows) & (offs_n[None, :] < n_cols)
    x = tl.load(x_int8_ptr + offs_m[:, None] * n_cols + offs_n[None, :], mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < n_cols).to(tl.float32)
    zero = tl.load(zero_ptr + offs_n, mask=offs_n < n_cols).to(tl.float32)
    out = (x - zero[None, :]) * scale[None, :]
    tl.store(out_ptr + offs_m[:, None] * n_cols + offs_n[None, :], out, mask=mask)


# 3. Grouped quantization dequant (groups of G consecutive weights share scale/zero)
@triton.jit
def grouped_dequant_kernel(x_int8_ptr, scale_ptr, zero_ptr, out_ptr,
                            n, group_size,
                            BLOCK: tl.constexpr):
    """Dequantize with grouped scale: each group_size elements share a scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_int8_ptr + offs, mask=mask).to(tl.float32)
    # Compute group index for each element
    group_idx = offs // group_size
    scale = tl.load(scale_ptr + group_idx, mask=mask)
    zero = tl.load(zero_ptr + group_idx, mask=mask)
    out = (x - zero) * scale
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Quantize fp32 -> int8 (symmetric: round(x / scale))
@triton.jit
def quantize_kernel(x_ptr, scale_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Symmetric quantization: out = clamp(round(x / scale), -128, 127)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    scale = tl.load(scale_ptr)
    q = x / scale
    # Round to nearest (manual since tl.math.round may not exist)
    q_rounded = tl.where(q >= 0, (q + 0.5).to(tl.int32), (q - 0.5).to(tl.int32))
    # Clamp to int8 range
    q_clamped = tl.maximum(tl.minimum(q_rounded, 127), -128)
    tl.store(out_ptr + offs, q_clamped, mask=mask)


# 5. RMSNorm (used in LLaMA instead of LayerNorm)
@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, row_stride, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    """RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    # Compute RMS
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=0) / n_cols
    rms = tl.sqrt(mean_x2 + eps)
    # Normalize and scale
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    out = x / rms * w
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 6. SiLU activation (used in LLaMA MLP)
@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """SiLU(x) = x * sigmoid(x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Fused residual + RMSNorm (LLaMA-style pre-norm)
@triton.jit
def residual_rmsnorm_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr,
                             row_stride, n_cols, eps,
                             BLOCK_SIZE: tl.constexpr):
    """out = RMSNorm(x + residual) * weight."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * row_stride + offs, mask=mask, other=0.0)
    h = x + r
    # RMSNorm
    h2 = h * h
    mean_h2 = tl.sum(h2, axis=0) / n_cols
    rms = tl.sqrt(mean_h2 + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    out = h / rms * w
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 8. Rotary position embedding (RoPE, used in LLaMA attention)
@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len, head_dim,
                BLOCK_D: tl.constexpr):
    """Apply RoPE: rotate pairs of elements by position-dependent angle."""
    pos = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    half_dim = head_dim // 2
    mask = offs_d < half_dim
    # Load x[pos, :half_dim] and x[pos, half_dim:]
    x_first = tl.load(x_ptr + pos * head_dim + offs_d, mask=mask, other=0.0)
    x_second = tl.load(x_ptr + pos * head_dim + half_dim + offs_d, mask=mask, other=0.0)
    # Load cos and sin for this position
    cos_val = tl.load(cos_ptr + pos * half_dim + offs_d, mask=mask, other=1.0)
    sin_val = tl.load(sin_ptr + pos * half_dim + offs_d, mask=mask, other=0.0)
    # Apply rotation
    out_first = x_first * cos_val - x_second * sin_val
    out_second = x_first * sin_val + x_second * cos_val
    tl.store(out_ptr + pos * head_dim + offs_d, out_first, mask=mask)
    tl.store(out_ptr + pos * head_dim + half_dim + offs_d, out_second, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_w8a16_matmul():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps', dtype=torch.float32)
    # Simulate int8 weights (stored as int32 for Metal compatibility)
    b_int8 = torch.randint(-128, 127, (K, N), device='mps', dtype=torch.int32)
    scale = torch.tensor([0.01], device='mps', dtype=torch.float32)
    c = torch.zeros(M, N, device='mps')
    grid = (1, 1)
    w8a16_matmul_kernel[grid](a, b_int8, scale, c, M, N, K,
                               a.stride(0), a.stride(1),
                               b_int8.stride(0), b_int8.stride(1),
                               c.stride(0), c.stride(1),
                               BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    ref = a @ (b_int8.float() * 0.01)
    err = (c - ref).abs().max().item()
    return err < 0.5, err


def test_per_channel_dequant():
    M, N = 32, 64
    x_int8 = torch.randint(-128, 127, (M, N), device='mps', dtype=torch.int32)
    scale = torch.rand(N, device='mps') * 0.1 + 0.01
    zero = torch.randint(-5, 5, (N,), device='mps').float()
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    per_channel_dequant_kernel[grid](x_int8, scale, zero, out, M, N,
                                      BLOCK_M=32, BLOCK_N=32)
    ref = (x_int8.float() - zero[None, :]) * scale[None, :]
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_grouped_dequant():
    n = 1024
    group_size = 128
    n_groups = n // group_size
    x_int8 = torch.randint(-128, 127, (n,), device='mps', dtype=torch.int32)
    scale = torch.rand(n_groups, device='mps') * 0.1 + 0.01
    zero = torch.randint(-5, 5, (n_groups,), device='mps').float()
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    grouped_dequant_kernel[grid](x_int8, scale, zero, out, n, group_size, BLOCK=256)
    # Reference
    group_idx = torch.arange(n, device='mps') // group_size
    ref = (x_int8.float() - zero[group_idx]) * scale[group_idx]
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_quantize():
    n = 1024
    x = torch.randn(n, device='mps') * 10
    scale = torch.tensor([x.abs().max().item() / 127.0], device='mps')
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    quantize_kernel[grid](x, scale, out, n, BLOCK=256)
    ref = (x / scale).round().clamp(-128, 127).int()
    err = (out - ref).abs().max().item()
    return err <= 1, err  # allow rounding difference of 1


def test_rmsnorm():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    rmsnorm_kernel[(M,)](x, w, out, N, N, eps, BLOCK_SIZE=128)
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


def test_residual_rmsnorm():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    r = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    residual_rmsnorm_kernel[(M,)](x, r, w, out, N, N, eps, BLOCK_SIZE=128)
    h = x + r
    rms = (h.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    ref = h / rms * w
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_rope():
    seq_len = 16
    head_dim = 64
    half_dim = head_dim // 2
    x = torch.randn(seq_len, head_dim, device='mps')
    # Position-dependent cos/sin
    pos = torch.arange(seq_len, device='mps').float()
    freq = 1.0 / (10000.0 ** (torch.arange(half_dim, device='mps').float() / half_dim))
    angles = pos[:, None] * freq[None, :]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    out = torch.zeros(seq_len, head_dim, device='mps')
    rope_kernel[(seq_len,)](x, cos_vals, sin_vals, out, seq_len, head_dim, BLOCK_D=32)
    # Reference
    x1 = x[:, :half_dim]
    x2 = x[:, half_dim:]
    ref = torch.cat([x1 * cos_vals - x2 * sin_vals,
                     x1 * sin_vals + x2 * cos_vals], dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 12) ===\n")
    print("   (Quantized matmul & LLM inference building blocks)\n")

    tests = [
        ("W8A16 Matmul (int8 weights)", test_w8a16_matmul),
        ("Per-Channel Dequant", test_per_channel_dequant),
        ("Grouped Dequant (group=128)", test_grouped_dequant),
        ("Quantize (fp32->int8)", test_quantize),
        ("RMSNorm (LLaMA)", test_rmsnorm),
        ("SiLU Activation", test_silu),
        ("Residual+RMSNorm (fused)", test_residual_rmsnorm),
        ("RoPE (Rotary Embedding)", test_rope),
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
