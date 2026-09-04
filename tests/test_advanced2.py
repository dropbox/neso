#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 2."""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Column-wise reduction (axis=0)
@triton.jit
def col_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum each column of a 2D matrix."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    col_sums = tl.sum(x, axis=0)  # reduce rows -> 1D
    tl.store(out_ptr + offs_n, col_sums, mask=offs_n < N)


# 2. Row-wise max reduction
@triton.jit
def row_max_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    row_maxs = tl.max(x, axis=1)
    tl.store(out_ptr + offs_m, row_maxs, mask=offs_m < M)


# 3. 2D element-wise add (load+add+store with 2D tiles)
@triton.jit
def add_2d_kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + ptrs, mask=mask)
    y = tl.load(y_ptr + ptrs, mask=mask)
    tl.store(out_ptr + ptrs, x + y, mask=mask)


# 4. RMSNorm (2D load, axis reduction, broadcast multiply)
@triton.jit
def rmsnorm_kernel(x_ptr, out_ptr, weight_ptr, row_stride, n_cols, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = x * rstd
    w = tl.load(weight_ptr + offs, mask=mask)
    out = xn * w
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 5. Fused softmax with causal mask (scf.if + 2D patterns)
@triton.jit
def causal_softmax_kernel(output_ptr, input_ptr, row_stride, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols
    x = tl.load(input_ptr + row_idx * row_stride + col_offs, mask=mask, other=-float('inf'))
    # Apply causal mask: zero out future positions
    causal_mask = col_offs <= row_idx
    x = tl.where(causal_mask, x, -float('inf'))
    # Softmax
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(output_ptr + row_idx * row_stride + col_offs, out, mask=mask)


# 6. Histogram (atomic add to bins)
@triton.jit
def histogram_kernel(x_ptr, hist_ptr, n, n_bins, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Bin values: clamp to [0, n_bins-1]
    bins = tl.minimum(tl.maximum(x, 0), n_bins - 1)
    bins_i32 = bins.to(tl.int32)
    # Atomic add to histogram
    tl.atomic_add(hist_ptr + bins_i32, 1, mask=mask)


# 7. Embedding lookup (gather from 2D table using 1D indices)
@triton.jit
def embedding_kernel(table_ptr, idx_ptr, out_ptr, n_indices, embed_dim,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_indices
    idx = tl.load(idx_ptr + offs, mask=mask)
    # Load embedding vectors
    for d in range(embed_dim):
        val = tl.load(table_ptr + idx * embed_dim + d, mask=mask)
        tl.store(out_ptr + offs * embed_dim + d, val, mask=mask)


# 8. Matmul with non-square shapes
@triton.jit
def matmul_ns_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


# 9. Cosine similarity (combines reduction, sqrt, division)
@triton.jit
def cosine_sim_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    dot_xy = tl.sum(x * y, axis=0)
    norm_x = tl.sqrt(tl.sum(x * x, axis=0))
    norm_y = tl.sqrt(tl.sum(y * y, axis=0))
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, dot_xy / (norm_x * norm_y + 1e-8))


# 10. Fused bias + activation (2D load, 1D bias broadcast, activation)
@triton.jit
def fused_bias_relu_kernel(x_ptr, bias_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    # Load bias (1D) and broadcast to 2D
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    # Add bias (broadcast) and apply ReLU
    out = tl.where(x + bias[:, None].T > 0, x + bias[:, None].T, 0.0)
    # Actually, simpler: just use per-element ops
    tl.store(x_ptrs, out, mask=mask)  # in-place


# ============================================================
# Test runners
# ============================================================

def test_col_sum():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(N, device='mps')
    col_sum_kernel[(triton.cdiv(N, 64),)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.sum(dim=0)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_row_max():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.full((M,), -float('inf'), device='mps')
    row_max_kernel[(triton.cdiv(M, 32),)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.max(dim=1).values
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_add_2d():
    M, N = 64, 64
    x = torch.randn(M, N, device='mps')
    y = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    add_2d_kernel[grid](x, y, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_rmsnorm():
    M, N = 16, 128
    x = torch.randn(M, N, device='mps')
    w = torch.ones(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    rmsnorm_kernel[(M,)](x, out, w, N, N, 1e-5, BLOCK_SIZE=128)
    # Compute reference
    ms = (x ** 2).mean(dim=1, keepdim=True)
    ref = x / torch.sqrt(ms + 1e-5) * w
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_causal_softmax():
    M = 16
    N = 16
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    causal_softmax_kernel[(M,)](out, x, N, N, BLOCK_SIZE=16)
    # Compute reference
    causal_mask = torch.tril(torch.ones(M, N, device='mps')).bool()
    x_masked = x.clone()
    x_masked[~causal_mask] = -float('inf')
    ref = torch.softmax(x_masked, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_histogram():
    n = 4096
    n_bins = 10
    x = torch.randint(0, n_bins, (n,), device='mps', dtype=torch.int32)
    hist = torch.zeros(n_bins, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    histogram_kernel[grid](x, hist, n, n_bins, BLOCK=256)
    ref = torch.zeros(n_bins, device='mps', dtype=torch.int32)
    for i in range(n_bins):
        ref[i] = (x == i).sum()
    err = (hist - ref).abs().max().item()
    return err == 0, err


def test_embedding():
    vocab_size = 100
    embed_dim = 4  # small for loop-based kernel
    n_indices = 256
    table = torch.randn(vocab_size, embed_dim, device='mps')
    idx = torch.randint(0, vocab_size, (n_indices,), device='mps', dtype=torch.int32)
    out = torch.zeros(n_indices, embed_dim, device='mps')
    grid = lambda meta: (triton.cdiv(n_indices, meta['BLOCK']),)
    embedding_kernel[grid](table, idx, out, n_indices, embed_dim, BLOCK=256)
    ref = table[idx.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_matmul_nonsquare():
    M, N, K = 64, 32, 64
    BM, BN, BK = 32, 32, 32
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_ns_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    ref = a @ b
    err = (c - ref).abs().max().item()
    tol = 1e-3 * (K ** 0.5)
    return err < tol, err


def test_cosine_sim():
    n = 256
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    cosine_sim_kernel[(1,)](x, y, out, n, BLOCK=256)
    ref = torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    err = abs(out.item() - ref.item())
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 2) ===\n")

    tests = [
        ("Col Sum (2D reduce axis=0)", test_col_sum),
        ("Row Max (2D reduce axis=1, max)", test_row_max),
        ("2D Element-wise Add", test_add_2d),
        ("RMSNorm", test_rmsnorm),
        ("Causal Softmax (tl.where mask)", test_causal_softmax),
        ("Histogram (atomic_add)", test_histogram),
        ("Embedding Lookup", test_embedding),
        ("Matmul Non-Square (64x32x64)", test_matmul_nonsquare),
        ("Cosine Similarity", test_cosine_sim),
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
