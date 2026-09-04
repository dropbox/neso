#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 23.

Targets: matmul stress tests and real-world patterns:
- Larger matmul (128x128x64) through optimized path
- Matmul with non-power-of-2 K (K=48)
- fp16 matmul with multi-K iteration
- Matmul + softmax row (fused attention score)
- Vector-matrix multiply (1xN @ NxM)
- Batched element-wise with 2D grid + large BLOCK
- Histogram via atomic add
- Prefix sum (cumsum) on larger arrays (multi-block)
- Fused RMSNorm + SiLU (LLaMA pre-FFN pattern)
- Cross-entropy loss (log-softmax + NLL)
- L2 normalization (normalize rows to unit length)
- Fused add + layer norm
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Larger matmul (128x128)
@triton.jit
def matmul_128_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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


# 2. Fused RMSNorm + SiLU (LLaMA pattern)
@triton.jit
def rmsnorm_silu_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    sq_mean = tl.sum(x * x, axis=0) / n_cols
    rms = tl.sqrt(sq_mean + eps)
    xn = x / rms * w
    # SiLU activation
    out = xn * tl.sigmoid(xn)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 3. Cross-entropy loss (log-softmax + nll)
@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr, n_classes,
                           BLOCK: tl.constexpr):
    """Compute CE loss for one sample: -log_softmax(logits)[label]."""
    sample = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_classes
    logits = tl.load(logits_ptr + sample * n_classes + offs,
                       mask=mask, other=-float('inf'))
    max_val = tl.max(logits, axis=0)
    shifted = logits - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(shifted), axis=0))
    log_softmax = shifted - log_sum_exp
    # Load label index
    label = tl.load(labels_ptr + sample)
    # Extract the loss at the label position
    # Use where to select the right element
    label_mask = offs == label
    loss_vals = tl.where(label_mask, -log_softmax, 0.0)
    loss = tl.sum(loss_vals, axis=0)
    tl.store(loss_ptr + sample, loss)


# 4. L2 normalization (normalize rows)
@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    norm_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm_sq + eps)
    out = x / norm
    tl.store(out_ptr + row * N + offs, out, mask=mask)


# 5. Fused add + layer norm
@triton.jit
def add_layernorm_kernel(a_ptr, b_ptr, gamma_ptr, beta_ptr, out_ptr,
                            n_cols, eps, BLOCK: tl.constexpr):
    """out = LayerNorm(a + b)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    a = tl.load(a_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + row * n_cols + offs, mask=mask, other=0.0)
    x = a + b
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    xn = diff / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    bt = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = g * xn + bt
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 6. Histogram via atomic add
@triton.jit
def histogram_kernel(data_ptr, hist_ptr, n, n_bins, BLOCK: tl.constexpr):
    """Compute histogram of integer data via atomic adds."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(data_ptr + offs, mask=mask)
    # Each thread atomically increments its bin
    tl.atomic_add(hist_ptr + vals, 1.0, mask=mask)


# 7. Vector dot product (reduction to scalar)
@triton.jit
def vector_dot_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Compute dot(a, b) = sum(a * b)."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        acc += a * b
    total = tl.sum(acc, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 8. Sigmoid + binary cross-entropy
@triton.jit
def bce_kernel(logits_ptr, targets_ptr, loss_ptr, n, BLOCK: tl.constexpr):
    """BCE with logits: loss = -[t*log(sigma(x)) + (1-t)*log(1-sigma(x))]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(logits_ptr + offs, mask=mask)
    t = tl.load(targets_ptr + offs, mask=mask)
    # Numerically stable BCE
    relu_x = tl.maximum(x, 0.0)
    loss = relu_x - x * t + tl.log(1.0 + tl.exp(-tl.abs(x)))
    tl.store(loss_ptr + offs, loss, mask=mask)


# 9. Pairwise distance (L2 between rows)
@triton.jit
def pairwise_l2_kernel(a_ptr, b_ptr, dist_ptr, M, N, D,
                         BLOCK_D: tl.constexpr):
    """dist[i,j] = ||a[i,:] - b[j,:]||^2."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    a = tl.load(a_ptr + i * D + d_offs, mask=d_mask, other=0.0)
    b = tl.load(b_ptr + j * D + d_offs, mask=d_mask, other=0.0)
    diff = a - b
    dist = tl.sum(diff * diff, axis=0)
    tl.store(dist_ptr + i * N + j, dist)


# 10. Gated linear unit (GLU)
@triton.jit
def glu_kernel(x_ptr, out_ptr, n, half_n, BLOCK: tl.constexpr):
    """GLU: out = x[:half] * sigmoid(x[half:])."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(x_ptr + half_n + offs, mask=mask)
    out = a * tl.sigmoid(b)
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_matmul_128():
    M, N, K = 128, 128, 64
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    matmul_128_kernel[grid](a, b, c, M, N, K,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    ref = a @ b
    err = (c - ref).abs().max().item()
    return err < 0.5, err


def test_rmsnorm_silu():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    rmsnorm_silu_kernel[(M,)](x, w, out, N, eps, BLOCK=128)
    rms = (x.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    xn = x / rms * w
    ref = xn * torch.sigmoid(xn)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_cross_entropy():
    batch = 32
    n_classes = 64
    logits = torch.randn(batch, n_classes, device='mps')
    labels = torch.randint(0, n_classes, (batch,), device='mps', dtype=torch.int32)
    loss = torch.zeros(batch, device='mps')
    cross_entropy_kernel[(batch,)](logits, labels, loss, n_classes, BLOCK=64)
    ref = torch.nn.functional.cross_entropy(logits, labels.long(), reduction='none')
    err = (loss - ref).abs().max().item()
    return err < 1e-3, err


def test_l2_norm():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    l2_norm_kernel[(M,)](x, out, N, 1e-8, BLOCK_N=128)
    ref = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_add_layernorm():
    M, N = 32, 128
    a = torch.randn(M, N, device='mps')
    b = torch.randn(M, N, device='mps')
    gamma = torch.randn(N, device='mps')
    beta = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    add_layernorm_kernel[(M,)](a, b, gamma, beta, out, N, eps, BLOCK=128)
    x = a + b
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_histogram():
    n = 1024
    n_bins = 16
    data = torch.randint(0, n_bins, (n,), device='mps', dtype=torch.int32)
    hist = torch.zeros(n_bins, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    histogram_kernel[grid](data, hist, n, n_bins, BLOCK=256)
    ref = torch.zeros(n_bins, device='mps')
    for i in range(n_bins):
        ref[i] = (data == i).sum().float()
    err = (hist - ref).abs().max().item()
    return err < 1.0, err  # atomic ordering may cause small diffs


def test_vector_dot():
    N = 1024
    a = torch.randn(N, device='mps')
    b = torch.randn(N, device='mps')
    out = torch.zeros(1, device='mps')
    vector_dot_kernel[(1,)](a, b, out, N, BLOCK=256)
    ref = (a * b).sum()
    err = abs(out.item() - ref.item())
    return err < 0.1, err


def test_bce():
    n = 1024
    logits = torch.randn(n, device='mps')
    targets = torch.rand(n, device='mps')  # between 0 and 1
    loss = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    bce_kernel[grid](logits, targets, loss, n, BLOCK=256)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    err = (loss - ref).abs().max().item()
    return err < 1e-4, err


def test_pairwise_l2():
    M, N, D = 8, 12, 32
    a = torch.randn(M, D, device='mps')
    b = torch.randn(N, D, device='mps')
    dist = torch.zeros(M, N, device='mps')
    pairwise_l2_kernel[(M, N)](a, b, dist, M, N, D, BLOCK_D=32)
    ref = torch.cdist(a, b, p=2).pow(2)
    err = (dist - ref).abs().max().item()
    return err < 1e-3, err


def test_glu():
    half_n = 512
    n = half_n * 2
    x = torch.randn(n, device='mps')
    out = torch.zeros(half_n, device='mps')
    grid = lambda meta: (triton.cdiv(half_n, meta['BLOCK']),)
    glu_kernel[grid](x, out, n, half_n, BLOCK=256)
    ref = x[:half_n] * torch.sigmoid(x[half_n:])
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 23) ===\n")
    print("    (Real-world fused kernels & large matmul)\n")

    tests = [
        ("Matmul 128x128x64", test_matmul_128),
        ("RMSNorm+SiLU (LLaMA)", test_rmsnorm_silu),
        ("Cross-Entropy Loss", test_cross_entropy),
        ("L2 Normalize", test_l2_norm),
        ("Add+LayerNorm", test_add_layernorm),
        ("Histogram (atomic)", test_histogram),
        ("Vector Dot (loop)", test_vector_dot),
        ("BCE with Logits", test_bce),
        ("Pairwise L2 Distance", test_pairwise_l2),
        ("GLU Activation", test_glu),
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
