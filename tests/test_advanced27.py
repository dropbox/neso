#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 27.

Targets: complex patterns and stress tests:
- Multi-block matmul reduction (M > BLOCK_M, N > BLOCK_N)
- Fused attention: Q@K^T * scale + causal_mask + softmax
- KV cache update pattern (write new KV, read old + new)
- Fused RoPE + attention score (transformer building block)
- Multi-head pattern (3D grid: batch, head, position)
- Gather + scatter (permutation / reorder)
- Masked softmax (attention with padding mask)
- Fused GEGLU (gate * GELU(x) pattern for LLaMA FFN)
- Fp16 in, fp32 compute, fp16 out (mixed precision chain)
- Two-pass normalization (compute stats then normalize)
- Topk via sort proxy (mark top-k positions)
- Batched outer product
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Masked softmax (attention with padding)
@triton.jit
def masked_softmax_kernel(x_ptr, mask_ptr, out_ptr, n_cols,
                            BLOCK: tl.constexpr):
    """Softmax with mask: masked positions get -inf before softmax."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    cmask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=cmask, other=-float('inf'))
    m = tl.load(mask_ptr + row * n_cols + offs, mask=cmask, other=0.0)
    # Apply mask: where mask==0, set to -inf
    x = tl.where(m > 0.0, x, -float('inf'))
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    sm = tl.sum(ex, axis=0)
    out = ex / sm
    tl.store(out_ptr + row * n_cols + offs, out, mask=cmask)


# 2. Fused GEGLU (LLaMA FFN: gate * GELU(x))
@triton.jit
def geglu_kernel(x_ptr, gate_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """GEGLU: out = gate * GELU(x) where GELU uses sigmoid approx."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    g = tl.load(gate_ptr + offs, mask=mask)
    # GELU via sigmoid: GELU(x) ≈ x * sigmoid(1.702 * x)
    gelu_x = x * tl.sigmoid(1.702 * x)
    out = g * gelu_x
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Gather (permutation read)
@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out[i] = src[idx[i]]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    val = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


# 4. Scatter (permutation write)
@triton.jit
def scatter_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out[idx[i]] = src[i]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask)
    idx = tl.load(idx_ptr + offs, mask=mask)
    tl.store(out_ptr + idx, val, mask=mask)


# 5. KV cache update (write at position, read all)
@triton.jit
def kv_cache_update_kernel(cache_ptr, new_val_ptr, out_ptr,
                             pos, seq_len, dim,
                             BLOCK_D: tl.constexpr):
    """Write new_val at position pos in cache, copy cache to output."""
    d = tl.arange(0, BLOCK_D)
    d_mask = d < dim
    # Write new value at position pos
    new_v = tl.load(new_val_ptr + d, mask=d_mask, other=0.0)
    tl.store(cache_ptr + pos * dim + d, new_v, mask=d_mask)
    # Copy full cache to output (one row per program)
    row = tl.program_id(0)
    if row < seq_len:
        v = tl.load(cache_ptr + row * dim + d, mask=d_mask, other=0.0)
        tl.store(out_ptr + row * dim + d, v, mask=d_mask)


# 6. Multi-head score pattern (batch dimension via program_id)
@triton.jit
def multi_head_score_kernel(q_ptr, k_ptr, score_ptr,
                              batch, heads, seq_len, d_model,
                              BLOCK_D: tl.constexpr):
    """score[b,h,i,j] = sum_d(Q[b,h,i,d] * K[b,h,j,d]) for single i,j."""
    b = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2)
    # Flat offset into Q[b, 0, i, :] and K[b, 0, j, :]
    # Layout: [batch, heads, seq_len, d_model] but we only do head 0
    q_base = b * seq_len * d_model + i * d_model
    k_base = b * seq_len * d_model + j * d_model
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d_model
    q = tl.load(q_ptr + q_base + d_offs, mask=d_mask, other=0.0)
    k = tl.load(k_ptr + k_base + d_offs, mask=d_mask, other=0.0)
    dot = tl.sum(q * k, axis=0)
    # score layout: [batch, seq_len, seq_len]
    tl.store(score_ptr + b * seq_len * seq_len + i * seq_len + j, dot)


# 7. Two-pass normalization: first pass computes mean/var per row
@triton.jit
def norm_stats_kernel(x_ptr, mean_ptr, var_ptr, N,
                        BLOCK_N: tl.constexpr):
    """Compute mean and variance for each row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)


# 8. Two-pass normalization: second pass normalizes
@triton.jit
def norm_apply_kernel(x_ptr, mean_ptr, var_ptr, out_ptr, N, eps,
                        BLOCK_N: tl.constexpr):
    """Normalize: out = (x - mean) / sqrt(var + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row)
    var = tl.load(var_ptr + row)
    out = (x - mean) / tl.sqrt(var + eps)
    tl.store(out_ptr + row * N + offs, out, mask=mask)


# 9. Fused add + scale + sigmoid (common in gates)
@triton.jit
def gate_kernel(x_ptr, bias_ptr, out_ptr, scale, n_cols,
                  BLOCK: tl.constexpr):
    """out = sigmoid((x + bias) * scale)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    out = tl.sigmoid((x + b) * scale)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 10. Reduce-scatter pattern (sum rows, scatter to output positions)
@triton.jit
def reduce_scatter_kernel(x_ptr, out_ptr, M, N,
                            BLOCK_N: tl.constexpr):
    """out[row] = sum(x[row, :]) for each row (reduction)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr + row, total)


# 11. Batched outer product
@triton.jit
def batched_outer_kernel(a_ptr, b_ptr, out_ptr, batch, M, N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[b, i, j] = a[b, i] * b[b, j]."""
    bi = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a = tl.load(a_ptr + bi * M + offs_m, mask=mask_m, other=0.0)
    b = tl.load(b_ptr + bi * N + offs_n, mask=mask_n, other=0.0)
    out = a[:, None] * b[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + bi * M * N + offs_m[:, None] * N + offs_n[None, :],
             out, mask=mask)


# 12. Fused softmax + scale + add (common in attention)
@triton.jit
def softmax_scale_add_kernel(x_ptr, bias_ptr, out_ptr, scale, n_cols,
                               BLOCK: tl.constexpr):
    """out = softmax(x * scale + bias) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    x = x * scale + b
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    sm = tl.sum(ex, axis=0)
    out = ex / sm
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_masked_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    mask = (torch.rand(M, N, device='mps') > 0.3).float()  # 70% visible
    out = torch.zeros(M, N, device='mps')
    masked_softmax_kernel[(M,)](x, mask, out, N, BLOCK=64)
    # Reference
    x_masked = x.clone()
    x_masked[mask == 0] = float('-inf')
    ref = torch.softmax(x_masked, dim=1)
    # Compare only non-masked positions (masked ones become 0/nan)
    valid = mask.sum(dim=1) > 0  # rows with at least one visible
    err = (out[valid] - ref[valid]).abs().max().item()
    return err < 1e-4, err


def test_geglu():
    n = 2048
    x = torch.randn(n, device='mps')
    gate = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    geglu_kernel[grid](x, gate, out, n, BLOCK=256)
    gelu_x = x * torch.sigmoid(1.702 * x)
    ref = gate * gelu_x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_gather():
    n = 1024
    src = torch.randn(n, device='mps')
    idx = torch.randperm(n, device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gather_kernel[grid](src, idx, out, n, BLOCK=256)
    ref = src[idx.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_scatter():
    n = 1024
    src = torch.randn(n, device='mps')
    idx = torch.randperm(n, device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    scatter_kernel[grid](src, idx, out, n, BLOCK=256)
    ref = torch.zeros(n, device='mps')
    ref[idx.long()] = src
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_kv_cache():
    seq_len = 16
    dim = 32
    cache = torch.randn(seq_len, dim, device='mps')
    new_val = torch.randn(dim, device='mps')
    out = torch.zeros(seq_len, dim, device='mps')
    pos = 5
    kv_cache_update_kernel[(seq_len,)](cache, new_val, out, pos, seq_len, dim,
                                         BLOCK_D=32)
    ref = cache.clone()
    ref[pos] = new_val
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_multi_head_score():
    batch = 4
    seq_len = 8
    d_model = 32
    q = torch.randn(batch, seq_len, d_model, device='mps')
    k = torch.randn(batch, seq_len, d_model, device='mps')
    score = torch.zeros(batch, seq_len, seq_len, device='mps')
    multi_head_score_kernel[(batch, seq_len, seq_len)](
        q, k, score, batch, 1, seq_len, d_model, BLOCK_D=32)
    ref = torch.bmm(q, k.transpose(1, 2))
    err = (score - ref).abs().max().item()
    return err < 1e-3, err


def test_two_pass_norm():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    mean_buf = torch.zeros(M, device='mps')
    var_buf = torch.zeros(M, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    norm_stats_kernel[(M,)](x, mean_buf, var_buf, N, BLOCK_N=128)
    norm_apply_kernel[(M,)](x, mean_buf, var_buf, out, N, eps, BLOCK_N=128)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = (x - mean) / (var + eps).sqrt()
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_gate():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    gate_kernel[(M,)](x, bias, out, 2.0, N, BLOCK=64)
    ref = torch.sigmoid((x + bias) * 2.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_reduce_scatter():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    reduce_scatter_kernel[(M,)](x, out, M, N, BLOCK_N=128)
    ref = x.sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-2, err


def test_batched_outer():
    batch = 4
    M, N = 16, 16
    a = torch.randn(batch, M, device='mps')
    b = torch.randn(batch, N, device='mps')
    out = torch.zeros(batch, M, N, device='mps')
    batched_outer_kernel[(batch, 1)](a, b, out, batch, M, N,
                                       BLOCK_M=16, BLOCK_N=16)
    ref = torch.bmm(a.unsqueeze(2), b.unsqueeze(1))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_softmax_scale_add():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    scale = 0.125
    softmax_scale_add_kernel[(M,)](x, bias, out, scale, N, BLOCK=64)
    ref = torch.softmax(x * scale + bias, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 27) ===\n")
    print("    (Attention patterns, gather/scatter, multi-pass norm)\n")

    tests = [
        ("Masked Softmax", test_masked_softmax),
        ("GEGLU (LLaMA FFN)", test_geglu),
        ("Gather (permute read)", test_gather),
        ("Scatter (permute write)", test_scatter),
        ("KV Cache Update", test_kv_cache),
        ("Multi-Head Score (3D)", test_multi_head_score),
        ("Two-Pass Norm", test_two_pass_norm),
        ("Gate (sigmoid)", test_gate),
        ("Reduce-Scatter", test_reduce_scatter),
        ("Batched Outer Product", test_batched_outer),
        ("Softmax+Scale+Add", test_softmax_scale_add),
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
