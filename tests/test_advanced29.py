#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 29.

Targets: LLM inference building blocks and edge cases:
- Fused RoPE (rotary position embedding, realistic)
- Token mixing (weighted sum of embeddings)
- Fused softmax + top-p (nucleus sampling prep)
- Cumulative softmax (online softmax via loop)
- Row-wise L1 normalization
- Fused bias + SiLU (LLaMA FFN gate)
- Multi-query attention score (shared K across heads)
- Fused add + clamp (gradient clipping pattern)
- Row-wise median approximation (via sort proxy)
- Fp16 chain (load fp16, compute fp32, store fp16)
- Matrix-vector product (Mx1 output via 2D tile)
- Fused RMSNorm + residual
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Fused RoPE (simplified, single head)
@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len, d_model,
                  BLOCK_D: tl.constexpr):
    """Apply RoPE: out[pos, 2k] = x[pos,2k]*cos - x[pos,2k+1]*sin
                    out[pos, 2k+1] = x[pos,2k]*sin + x[pos,2k+1]*cos."""
    pos = tl.program_id(0)
    half_d = d_model // 2
    k_offs = tl.arange(0, BLOCK_D)
    k_mask = k_offs < half_d
    # Load pairs
    x_even = tl.load(x_ptr + pos * d_model + k_offs * 2, mask=k_mask, other=0.0)
    x_odd = tl.load(x_ptr + pos * d_model + k_offs * 2 + 1, mask=k_mask, other=0.0)
    cos_val = tl.load(cos_ptr + pos * half_d + k_offs, mask=k_mask, other=1.0)
    sin_val = tl.load(sin_ptr + pos * half_d + k_offs, mask=k_mask, other=0.0)
    # Apply rotation
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd = x_even * sin_val + x_odd * cos_val
    tl.store(out_ptr + pos * d_model + k_offs * 2, out_even, mask=k_mask)
    tl.store(out_ptr + pos * d_model + k_offs * 2 + 1, out_odd, mask=k_mask)


# 2. Row-wise L1 normalization
@triton.jit
def l1_norm_kernel(x_ptr, out_ptr, N, eps, BLOCK_N: tl.constexpr):
    """out[row,:] = x[row,:] / (sum(|x[row,:]|) + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    l1 = tl.sum(tl.abs(x), axis=0) + eps
    out = x / l1
    tl.store(out_ptr + row * N + offs, out, mask=mask)


# 3. Fused bias + SiLU (LLaMA FFN gate)
@triton.jit
def bias_silu_kernel(x_ptr, bias_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    """out = SiLU(x + bias) = (x + bias) * sigmoid(x + bias)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    xb = x + b
    out = xb * tl.sigmoid(xb)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 4. Fused add + clamp (gradient clipping pattern)
@triton.jit
def add_clamp_kernel(x_ptr, y_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
    """out = clamp(x + y, lo, hi)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = tl.minimum(tl.maximum(x + y, lo), hi)
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Fp16 load -> fp32 compute -> fp16 store
@triton.jit
def fp16_chain_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Load fp16, compute in fp32, store fp16: out = sigmoid(x + y)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    out = tl.sigmoid(x + y)
    tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)


# 6. Fused RMSNorm + residual
@triton.jit
def rmsnorm_residual_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr,
                               n_cols, eps, BLOCK: tl.constexpr):
    """out = RMSNorm(x) * weight + residual."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * n_cols + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    # RMS norm
    ms = tl.sum(x * x, axis=0) / n_cols
    rms = tl.rsqrt(ms + eps)
    out = x * rms * w + r
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 7. Negation + abs (simple but tests neg)
@triton.jit
def neg_abs_kernel(x_ptr, neg_ptr, abs_ptr, n, BLOCK: tl.constexpr):
    """neg = -x, abs_out = |x|."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(neg_ptr + offs, -x, mask=mask)
    tl.store(abs_ptr + offs, tl.abs(x), mask=mask)


# 8. Fused multiply-add chain (a*b + c*d + e)
@triton.jit
def fma_chain_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, out_ptr, n,
                       BLOCK: tl.constexpr):
    """out = a*b + c*d + e."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    e = tl.load(e_ptr + offs, mask=mask)
    out = a * b + c * d + e
    tl.store(out_ptr + offs, out, mask=mask)


# 9. Row-wise sum of squares (for gradient norm computation)
@triton.jit
def sum_sq_kernel(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    """out[row] = sum(x[row,:]^2)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    sq = x * x
    total = tl.sum(sq, axis=0)
    tl.store(out_ptr + row, total)


# 10. Fused scale + add + sigmoid (logistic regression output)
@triton.jit
def logistic_kernel(x_ptr, w_ptr, b, out_ptr, n, BLOCK: tl.constexpr):
    """out = sigmoid(x * w + b)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    out = tl.sigmoid(x * w + b)
    tl.store(out_ptr + offs, out, mask=mask)


# 11. Token mixing: weighted average of embeddings
@triton.jit
def token_mix_kernel(emb_ptr, w_ptr, out_ptr, n_tokens, dim,
                       BLOCK_D: tl.constexpr, N_TOKENS: tl.constexpr):
    """out[d] = sum_t(w[t] * emb[t, d]) — weighted average over tokens."""
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(0, N_TOKENS):
        w_t = tl.load(w_ptr + t)
        emb = tl.load(emb_ptr + t * dim + d_offs, mask=d_mask, other=0.0)
        acc += w_t * emb
    tl.store(out_ptr + d_offs, acc, mask=d_mask)


# 12. Clipped ReLU (ReLU6)
@triton.jit
def relu6_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = min(max(x, 0), 6)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_rope():
    seq_len = 32
    d_model = 64
    half_d = d_model // 2
    x = torch.randn(seq_len, d_model, device='mps')
    # Generate cos/sin tables
    pos = torch.arange(seq_len, device='mps').float()
    freqs = 1.0 / (10000 ** (torch.arange(0, half_d, device='mps').float() / half_d))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [seq, half_d]
    cos_tab = torch.cos(angles)
    sin_tab = torch.sin(angles)
    out = torch.zeros_like(x)
    rope_kernel[(seq_len,)](x, cos_tab, sin_tab, out, seq_len, d_model,
                              BLOCK_D=32)
    # Reference
    ref = torch.zeros_like(x)
    ref[:, 0::2] = x[:, 0::2] * cos_tab - x[:, 1::2] * sin_tab
    ref[:, 1::2] = x[:, 0::2] * sin_tab + x[:, 1::2] * cos_tab
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_l1_norm():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-8
    l1_norm_kernel[(M,)](x, out, N, eps, BLOCK_N=64)
    ref = x / (x.abs().sum(dim=1, keepdim=True) + eps)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_bias_silu():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    bias_silu_kernel[(M,)](x, bias, out, N, BLOCK=128)
    xb = x + bias
    ref = xb * torch.sigmoid(xb)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_add_clamp():
    n = 2048
    x = torch.randn(n, device='mps') * 5
    y = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_clamp_kernel[grid](x, y, out, -1.0, 1.0, n, BLOCK=256)
    ref = (x + y).clamp(-1.0, 1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_fp16_chain():
    n = 2048
    x = torch.randn(n, device='mps', dtype=torch.float16)
    y = torch.randn(n, device='mps', dtype=torch.float16)
    out = torch.zeros(n, device='mps', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fp16_chain_kernel[grid](x, y, out, n, BLOCK=256)
    ref = torch.sigmoid((x.float() + y.float())).half()
    err = (out.float() - ref.float()).abs().max().item()
    return err < 2e-3, err  # fp16 has ~1e-3 precision


def test_rmsnorm_residual():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    residual = torch.randn(M, N, device='mps')
    weight = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    rmsnorm_residual_kernel[(M,)](x, residual, weight, out, N, eps, BLOCK=128)
    ms = (x * x).mean(dim=1, keepdim=True)
    rms = torch.rsqrt(ms + eps)
    ref = x * rms * weight + residual
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_neg_abs():
    n = 1024
    x = torch.randn(n, device='mps')
    neg = torch.zeros(n, device='mps')
    abs_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    neg_abs_kernel[grid](x, neg, abs_out, n, BLOCK=256)
    err = max((neg - (-x)).abs().max().item(),
              (abs_out - x.abs()).abs().max().item())
    return err < 1e-6, err


def test_fma_chain():
    n = 2048
    a, b, c, d, e = [torch.randn(n, device='mps') for _ in range(5)]
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fma_chain_kernel[grid](a, b, c, d, e, out, n, BLOCK=256)
    ref = a * b + c * d + e
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_sum_sq():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    sum_sq_kernel[(M,)](x, out, N, BLOCK_N=128)
    ref = (x * x).sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-2, err


def test_logistic():
    n = 2048
    x = torch.randn(n, device='mps')
    w = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    logistic_kernel[grid](x, w, 0.5, out, n, BLOCK=256)
    ref = torch.sigmoid(x * w + 0.5)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_token_mix():
    n_tokens = 8
    dim = 64
    emb = torch.randn(n_tokens, dim, device='mps')
    w = torch.randn(n_tokens, device='mps')
    out = torch.zeros(dim, device='mps')
    token_mix_kernel[(1,)](emb, w, out, n_tokens, dim,
                             BLOCK_D=64, N_TOKENS=8)
    ref = (w.unsqueeze(1) * emb).sum(dim=0)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_relu6():
    n = 2048
    x = torch.randn(n, device='mps') * 10
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    relu6_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.clamp(x, 0, 6)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 29) ===\n")
    print("    (LLM inference: RoPE, RMSNorm, fp16, token mix)\n")

    tests = [
        ("RoPE (rotary embed)", test_rope),
        ("L1 Normalize", test_l1_norm),
        ("Bias+SiLU (gate)", test_bias_silu),
        ("Add+Clamp", test_add_clamp),
        ("Fp16 Chain", test_fp16_chain),
        ("RMSNorm+Residual", test_rmsnorm_residual),
        ("Neg+Abs", test_neg_abs),
        ("FMA Chain (5 input)", test_fma_chain),
        ("Sum of Squares", test_sum_sq),
        ("Logistic (sigmoid)", test_logistic),
        ("Token Mixing", test_token_mix),
        ("ReLU6 (clipped)", test_relu6),
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
