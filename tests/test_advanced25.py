#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 25.

Targets: transformer inference patterns and edge cases:
- Multi-head attention score (Q@K^T per head, no V multiply)
- Fused softmax + scale (attention score normalization)
- Top-k masking (keep top-k values, zero rest)
- Strided batch matmul (3D indexing with program_id)
- Fused add-multiply-add (residual connection pattern)
- Row-wise argmax
- Exponential decay (geometric series)
- Chunked reduction (multi-block sum via atomics)
- Fused layer norm + residual
- Leaky ReLU with configurable slope
- Power function (x^n for integer n)
- Fused scale + bias + clip (quantization pre-processing)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Attention score: Q@K^T with scale (single head, small)
@triton.jit
def attn_score_kernel(q_ptr, k_ptr, score_ptr, seq_len, d_model,
                       scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """score[i,j] = sum_d(Q[i,d] * K[j,d]) * scale."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d_model
    q = tl.load(q_ptr + i * d_model + d_offs, mask=d_mask, other=0.0)
    k = tl.load(k_ptr + j * d_model + d_offs, mask=d_mask, other=0.0)
    dot = tl.sum(q * k, axis=0)
    tl.store(score_ptr + i * seq_len + j, dot * scale)


# 2. Fused softmax + scale
@triton.jit
def scaled_softmax_kernel(x_ptr, out_ptr, scale, n_cols,
                            BLOCK: tl.constexpr):
    """out = softmax(x * scale) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x = x * scale
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    sm = tl.sum(ex, axis=0)
    out = ex / sm
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 3. Fused add-multiply-add (residual pattern)
@triton.jit
def residual_fma_kernel(x_ptr, residual_ptr, scale_ptr, bias_ptr,
                          out_ptr, n, BLOCK: tl.constexpr):
    """out = (x + residual) * scale + bias (broadcast scale/bias)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(residual_ptr + offs, mask=mask)
    s = tl.load(scale_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    out = (x + r) * s + b
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Row-wise argmax (returns index of max per row)
@triton.jit
def argmax_kernel(x_ptr, idx_ptr, N, BLOCK_N: tl.constexpr):
    """idx[row] = argmax(x[row, :])."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
    best_idx = tl.argmax(x, axis=0)
    tl.store(idx_ptr + row, best_idx)


# 5. Chunked reduction via atomics (multi-block sum)
@triton.jit
def chunked_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Multi-block sum: each block sums its chunk, atomically adds to output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    chunk_sum = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, chunk_sum)


# 6. Fused layer norm + residual add
@triton.jit
def layernorm_residual_kernel(x_ptr, residual_ptr, gamma_ptr, beta_ptr,
                                out_ptr, n_cols, eps,
                                BLOCK: tl.constexpr):
    """out = LayerNorm(x) + residual."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    xn = diff / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = g * xn + b + r
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 7. Leaky ReLU
@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, slope, n, BLOCK: tl.constexpr):
    """out = x if x > 0 else slope * x."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.where(x > 0.0, x, x * slope)
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Power function (x^4 via repeated multiply)
@triton.jit
def pow4_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = x^4 = (x*x) * (x*x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    x2 = x * x
    out = x2 * x2
    tl.store(out_ptr + offs, out, mask=mask)


# 9. Fused scale + bias + clip (quantization pre-processing)
@triton.jit
def scale_bias_clip_kernel(x_ptr, out_ptr, scale, bias, lo, hi, n,
                             BLOCK: tl.constexpr):
    """out = clamp(x * scale + bias, lo, hi)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = x * scale + bias
    y = tl.minimum(tl.maximum(y, lo), hi)
    tl.store(out_ptr + offs, y, mask=mask)


# 10. Weighted sum of two tensors (linear interpolation / lerp)
@triton.jit
def lerp_kernel(a_ptr, b_ptr, out_ptr, t, n, BLOCK: tl.constexpr):
    """out = a * (1-t) + b * t."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = a * (1.0 - t) + b * t
    tl.store(out_ptr + offs, out, mask=mask)


# 11. Row-wise top-1 with value and index
@triton.jit
def topk1_kernel(x_ptr, val_ptr, idx_ptr, N, BLOCK_N: tl.constexpr):
    """Find max value and its index per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    tl.store(val_ptr + row, max_val)
    best_idx = tl.argmax(x, axis=0)
    tl.store(idx_ptr + row, best_idx)


# 12. Multi-output: compute exp, log, sqrt in one pass
@triton.jit
def multi_math_kernel(x_ptr, exp_ptr, log_ptr, sqrt_ptr, n,
                        BLOCK: tl.constexpr):
    """Compute exp(x), log(|x|+1), sqrt(|x|) in one kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    ax = tl.abs(x)
    tl.store(exp_ptr + offs, tl.exp(x), mask=mask)
    tl.store(log_ptr + offs, tl.log(ax + 1.0), mask=mask)
    tl.store(sqrt_ptr + offs, tl.sqrt(ax), mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_attn_score():
    seq_len = 32
    d_model = 64
    scale = 1.0 / (d_model ** 0.5)
    q = torch.randn(seq_len, d_model, device='mps')
    k = torch.randn(seq_len, d_model, device='mps')
    score = torch.zeros(seq_len, seq_len, device='mps')
    attn_score_kernel[(seq_len, seq_len)](q, k, score, seq_len, d_model,
                                            scale, BLOCK_S=32, BLOCK_D=64)
    ref = (q @ k.T) * scale
    err = (score - ref).abs().max().item()
    return err < 1e-3, err


def test_scaled_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    scale = 0.125
    scaled_softmax_kernel[(M,)](x, out, scale, N, BLOCK=64)
    ref = torch.softmax(x * scale, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_residual_fma():
    n = 2048
    x = torch.randn(n, device='mps')
    r = torch.randn(n, device='mps')
    s = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    residual_fma_kernel[grid](x, r, s, b, out, n, BLOCK=256)
    ref = (x + r) * s + b
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_argmax():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    idx = torch.zeros(M, device='mps', dtype=torch.int32)
    argmax_kernel[(M,)](x, idx, N, BLOCK_N=128)
    ref = x.argmax(dim=1).int()
    err = (idx - ref).abs().max().item()
    return err == 0, float(err)


def test_chunked_sum():
    n = 4096
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    chunked_sum_kernel[grid](x, out, n, BLOCK=256)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 0.5, err  # atomic adds can have ordering-dependent precision


def test_layernorm_residual():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    residual = torch.randn(M, N, device='mps')
    gamma = torch.randn(N, device='mps')
    beta = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    layernorm_residual_kernel[(M,)](x, residual, gamma, beta, out, N, eps,
                                      BLOCK=128)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta + residual
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_leaky_relu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    slope = 0.01
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    leaky_relu_kernel[grid](x, out, slope, n, BLOCK=256)
    ref = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_pow4():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    pow4_kernel[grid](x, out, n, BLOCK=256)
    ref = x ** 4
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_scale_bias_clip():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    scale_bias_clip_kernel[grid](x, out, 2.0, 0.5, -1.0, 1.0, n, BLOCK=256)
    ref = (x * 2.0 + 0.5).clamp(-1.0, 1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_lerp():
    n = 2048
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    t = 0.3
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    lerp_kernel[grid](a, b, out, t, n, BLOCK=256)
    ref = a * 0.7 + b * 0.3
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_topk1():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    val = torch.zeros(M, device='mps')
    idx = torch.zeros(M, device='mps', dtype=torch.int32)
    topk1_kernel[(M,)](x, val, idx, N, BLOCK_N=128)
    ref_val, ref_idx = x.max(dim=1)
    val_err = (val - ref_val).abs().max().item()
    idx_err = (idx - ref_idx.int()).abs().max().item()
    err = max(val_err, idx_err)
    return err < 1e-5, err


def test_multi_math():
    n = 1024
    x = torch.randn(n, device='mps') * 0.5  # keep values moderate for exp
    exp_out = torch.zeros(n, device='mps')
    log_out = torch.zeros(n, device='mps')
    sqrt_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_math_kernel[grid](x, exp_out, log_out, sqrt_out, n, BLOCK=256)
    ref_exp = torch.exp(x)
    ref_log = torch.log(x.abs() + 1.0)
    ref_sqrt = torch.sqrt(x.abs())
    err = max((exp_out - ref_exp).abs().max().item(),
              (log_out - ref_log).abs().max().item(),
              (sqrt_out - ref_sqrt).abs().max().item())
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 25) ===\n")
    print("    (Transformer inference, argmax, multi-math)\n")

    tests = [
        ("Attention Score (Q@K^T)", test_attn_score),
        ("Scaled Softmax", test_scaled_softmax),
        ("Residual FMA", test_residual_fma),
        ("Row Argmax", test_argmax),
        ("Chunked Sum (atomic)", test_chunked_sum),
        ("LayerNorm+Residual", test_layernorm_residual),
        ("Leaky ReLU", test_leaky_relu),
        ("Power x^4", test_pow4),
        ("Scale+Bias+Clip", test_scale_bias_clip),
        ("Lerp (interp)", test_lerp),
        ("Top-1 (val+idx)", test_topk1),
        ("Multi-Math (exp/log/sqrt)", test_multi_math),
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
