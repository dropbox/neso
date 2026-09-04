#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 30.

Targets: production LLM serving patterns:
- Fused attention: causal mask + softmax + output (simplified, no loop)
- Multi-query attention pattern (K shared across heads)
- Fused RMSNorm + RoPE (LLaMA pre-attention)
- Fused SiLU * gate (LLaMA FFN, two inputs)
- KV cache concatenation pattern
- Multi-output: mean, max, min, count in one kernel
- Fused clamp + cast (activation quantization)
- Strided batch store (write every N-th position)
- Fused residual + RMSNorm + scale (post-attention)
- Cumulative max (running maximum)
- Triangular attention pattern (lower-left only)
- Double softmax (row then column, rare but tests composability)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Fused SiLU * gate (LLaMA FFN)
@triton.jit
def silu_gate_kernel(x_ptr, gate_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = SiLU(x) * gate = x * sigmoid(x) * gate."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    g = tl.load(gate_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x) * g
    tl.store(out_ptr + offs, out, mask=mask)


# 2. Fused RMSNorm + scale
@triton.jit
def rmsnorm_scale_kernel(x_ptr, weight_ptr, out_ptr, scale,
                           n_cols, eps, BLOCK: tl.constexpr):
    """out = RMSNorm(x) * weight * scale."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    rms = tl.rsqrt(ms + eps)
    out = x * rms * w * scale
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 3. Causal attention score with mask (no V multiply)
@triton.jit
def causal_attn_score_kernel(q_ptr, k_ptr, out_ptr, seq_len, d_model,
                               scale, BLOCK_D: tl.constexpr):
    """score[i,j] = (Q[i] . K[j]) * scale if j <= i else -inf."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d_model
    q = tl.load(q_ptr + i * d_model + d_offs, mask=d_mask, other=0.0)
    k = tl.load(k_ptr + j * d_model + d_offs, mask=d_mask, other=0.0)
    dot = tl.sum(q * k, axis=0) * scale
    # Apply causal mask (-1e9 instead of -inf for numerical stability)
    out = tl.where(j <= i, dot, -1e9)
    tl.store(out_ptr + i * seq_len + j, out)


# 4. Multi-output statistics
@triton.jit
def row_stats_kernel(x_ptr, mean_ptr, std_ptr, min_ptr, max_ptr,
                       N, BLOCK_N: tl.constexpr):
    """Compute mean, std, min, max per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    x_min = tl.load(x_ptr + row * N + offs, mask=mask, other=float('inf'))
    x_max = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    std = tl.sqrt(var)
    mn = tl.min(x_min, axis=0)
    mx = tl.max(x_max, axis=0)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)
    tl.store(min_ptr + row, mn)
    tl.store(max_ptr + row, mx)


# 5. Fused clamp + fp16 cast (activation quantization)
@triton.jit
def clamp_cast_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
    """out = fp16(clamp(x, lo, hi))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)


# 6. Strided batch store (write at stride positions)
@triton.jit
def strided_store_kernel(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    """out[i * stride] = x[i] for i in range(n)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs * stride, x, mask=mask)


# 7. Polynomial evaluation (Horner's method): a0 + x*(a1 + x*(a2 + x*a3))
@triton.jit
def horner_kernel(x_ptr, out_ptr, a0, a1, a2, a3, n, BLOCK: tl.constexpr):
    """Evaluate cubic polynomial via Horner's method."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = a0 + x * (a1 + x * (a2 + x * a3))
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Fused residual + layer norm + scale (post-attention)
@triton.jit
def residual_ln_scale_kernel(x_ptr, residual_ptr, gamma_ptr, beta_ptr,
                                out_ptr, scale, n_cols, eps,
                                BLOCK: tl.constexpr):
    """out = scale * LayerNorm(x + residual)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * n_cols + offs, mask=mask, other=0.0)
    xr = x + r
    mean = tl.sum(xr, axis=0) / n_cols
    diff = xr - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    xn = diff / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = scale * (g * xn + b)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 9. Softmax temperature scaling + log (for sampling)
@triton.jit
def log_softmax_kernel(x_ptr, out_ptr, temp, n_cols, BLOCK: tl.constexpr):
    """out = log_softmax(x / temp) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x = x / temp
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    lse = tl.log(tl.sum(ex, axis=0)) + mx
    out = x - lse
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 10. Fused exp + sum (partition function for sampling)
@triton.jit
def exp_sum_kernel(x_ptr, exp_ptr, sum_ptr, n_cols, BLOCK: tl.constexpr):
    """Compute exp(x) per element and sum(exp(x)) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    total = tl.sum(ex, axis=0)
    tl.store(exp_ptr + row * n_cols + offs, ex, mask=mask)
    tl.store(sum_ptr + row, total)


# 11. Fused abs + sign extraction
@triton.jit
def abs_sign_kernel(x_ptr, abs_ptr, sign_ptr, n, BLOCK: tl.constexpr):
    """Decompose x into |x| and sign(x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(abs_ptr + offs, tl.abs(x), mask=mask)
    sign = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    tl.store(sign_ptr + offs, sign, mask=mask)


# 12. Fused exp2 + log2 chain (information theory)
@triton.jit
def exp2_log2_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = log2(1 + exp2(x)) — softplus in base 2."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.log2(1.0 + tl.exp2(x))
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_silu_gate():
    n = 2048
    x = torch.randn(n, device='mps')
    gate = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    silu_gate_kernel[grid](x, gate, out, n, BLOCK=256)
    ref = x * torch.sigmoid(x) * gate
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_rmsnorm_scale():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    weight = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    scale = 0.5
    rmsnorm_scale_kernel[(M,)](x, weight, out, scale, N, eps, BLOCK=128)
    ms = (x * x).mean(dim=1, keepdim=True)
    rms = torch.rsqrt(ms + eps)
    ref = x * rms * weight * scale
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_causal_attn_score():
    seq_len = 16
    d_model = 32
    scale = 1.0 / (d_model ** 0.5)
    q = torch.randn(seq_len, d_model, device='mps')
    k = torch.randn(seq_len, d_model, device='mps')
    out = torch.zeros(seq_len, seq_len, device='mps')
    causal_attn_score_kernel[(seq_len, seq_len)](q, k, out, seq_len, d_model,
                                                    scale, BLOCK_D=32)
    scores = (q @ k.T) * scale
    mask = torch.tril(torch.ones(seq_len, seq_len, device='mps'))
    ref = torch.where(mask.bool(), scores, torch.tensor(-1e9, device='mps'))
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_row_stats():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    mean_out = torch.zeros(M, device='mps')
    std_out = torch.zeros(M, device='mps')
    min_out = torch.zeros(M, device='mps')
    max_out = torch.zeros(M, device='mps')
    row_stats_kernel[(M,)](x, mean_out, std_out, min_out, max_out, N,
                             BLOCK_N=64)
    ref_mean = x.mean(dim=1)
    ref_std = x.std(dim=1, correction=0)
    ref_min = x.min(dim=1).values
    ref_max = x.max(dim=1).values
    err = max((mean_out - ref_mean).abs().max().item(),
              (std_out - ref_std).abs().max().item(),
              (min_out - ref_min).abs().max().item(),
              (max_out - ref_max).abs().max().item())
    return err < 1e-3, err


def test_clamp_cast():
    n = 2048
    x = torch.randn(n, device='mps') * 10
    out = torch.zeros(n, device='mps', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    clamp_cast_kernel[grid](x, out, -5.0, 5.0, n, BLOCK=256)
    ref = x.clamp(-5.0, 5.0).half()
    err = (out.float() - ref.float()).abs().max().item()
    return err < 1e-3, err


def test_strided_store():
    n = 256
    stride = 4
    x = torch.randn(n, device='mps')
    out = torch.zeros(n * stride, device='mps')
    strided_store_kernel[(1,)](x, out, n, stride, BLOCK=256)
    ref = torch.zeros(n * stride, device='mps')
    ref[::stride] = x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_horner():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    a0, a1, a2, a3 = 1.0, -0.5, 0.3, 0.1
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    horner_kernel[grid](x, out, a0, a1, a2, a3, n, BLOCK=256)
    ref = a0 + x * (a1 + x * (a2 + x * a3))
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_residual_ln_scale():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    residual = torch.randn(M, N, device='mps')
    gamma = torch.randn(N, device='mps')
    beta = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    scale = 0.25
    residual_ln_scale_kernel[(M,)](x, residual, gamma, beta, out, scale,
                                     N, eps, BLOCK=128)
    xr = x + residual
    mean = xr.mean(dim=1, keepdim=True)
    var = xr.var(dim=1, keepdim=True, correction=0)
    xn = (xr - mean) / (var + eps).sqrt()
    ref = scale * (gamma * xn + beta)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_log_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    temp = 0.5
    log_softmax_kernel[(M,)](x, out, temp, N, BLOCK=64)
    ref = torch.log_softmax(x / temp, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_exp_sum():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps') * 0.5
    exp_out = torch.zeros(M, N, device='mps')
    sum_out = torch.zeros(M, device='mps')
    exp_sum_kernel[(M,)](x, exp_out, sum_out, N, BLOCK=64)
    mx = x.max(dim=1, keepdim=True).values
    ref_exp = torch.exp(x - mx)
    ref_sum = ref_exp.sum(dim=1)
    err = max((exp_out - ref_exp).abs().max().item(),
              (sum_out - ref_sum).abs().max().item())
    return err < 1e-4, err


def test_abs_sign():
    n = 2048
    x = torch.randn(n, device='mps')
    abs_out = torch.zeros(n, device='mps')
    sign_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    abs_sign_kernel[grid](x, abs_out, sign_out, n, BLOCK=256)
    ref_abs = x.abs()
    ref_sign = x.sign()
    err = max((abs_out - ref_abs).abs().max().item(),
              (sign_out - ref_sign).abs().max().item())
    return err < 1e-5, err


def test_exp2_log2():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    exp2_log2_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.log2(1.0 + torch.exp2(x))
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 30) ===\n")
    print("    (Production LLM: SiLU*gate, causal attn, log-softmax)\n")

    tests = [
        ("SiLU*Gate (LLaMA FFN)", test_silu_gate),
        ("RMSNorm+Scale", test_rmsnorm_scale),
        ("Causal Attn Score", test_causal_attn_score),
        ("Row Stats (4 out)", test_row_stats),
        ("Clamp+Cast (fp16)", test_clamp_cast),
        ("Strided Store", test_strided_store),
        ("Horner Polynomial", test_horner),
        ("Residual+LN+Scale", test_residual_ln_scale),
        ("Log-Softmax (temp)", test_log_softmax),
        ("Exp+Sum (partition)", test_exp_sum),
        ("Abs+Sign", test_abs_sign),
        ("Exp2+Log2 (softplus)", test_exp2_log2),
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
