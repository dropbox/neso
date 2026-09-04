#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 6.

Targets: cross-iteration accumulators (non-matmul scf.for), strided access,
         mixed-type arithmetic, tl.where chains, multi-stage pipelines,
         indirect gather/scatter, online reduction, fused dropout.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Running sum via scf.for (cross-iteration accumulator, NOT matmul)
@triton.jit
def running_sum_kernel(x_ptr, out_ptr, n_chunks, chunk_size,
                       BLOCK: tl.constexpr):
    """Accumulate sums across chunks: out[i] = sum(x[0:chunk_size*(i+1)])"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_chunks):
        mask = offs < chunk_size
        x = tl.load(x_ptr + i * chunk_size + offs, mask=mask, other=0.0)
        acc += x
        # Store running sum after each chunk
        chunk_sum = tl.sum(acc, axis=0)
        if pid == 0:
            tl.store(out_ptr + i, chunk_sum)


# 2. Strided load/store (non-contiguous memory access)
@triton.jit
def strided_copy_kernel(x_ptr, out_ptr, n, stride_in, stride_out,
                        BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride_in, mask=mask)
    tl.store(out_ptr + offs * stride_out, x, mask=mask)


# 3. Fused GELU activation (compound math: x * 0.5 * (1 + erf(x/sqrt(2))))
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Approximate GELU: x * sigmoid(1.702 * x)
    out = x * tl.sigmoid(1.702 * x)
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Multi-stage pipeline: softmax + scale + clamp in one kernel
@triton.jit
def softmax_scale_clamp_kernel(x_ptr, out_ptr, scale, lo, hi, row_stride, n_cols,
                                BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols
    x = tl.load(x_ptr + row_idx * row_stride + col_offs, mask=mask, other=-float('inf'))
    # Softmax
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    sm = exp_x / sum_exp
    # Scale
    sm = sm * scale
    # Clamp
    sm = tl.minimum(tl.maximum(sm, lo), hi)
    tl.store(out_ptr + row_idx * row_stride + col_offs, sm, mask=mask)


# 5. Gather kernel (indirect load using index tensor)
@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    val = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


# 6. Scatter add kernel (indirect store with atomic)
@triton.jit
def scatter_add_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask)
    idx = tl.load(idx_ptr + offs, mask=mask)
    tl.atomic_add(out_ptr + idx, val, mask=mask)


# 7. Online max (streaming max over large array in chunks)
@triton.jit
def online_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Find global max across entire array."""
    offs = tl.arange(0, BLOCK)
    running_max = tl.load(x_ptr + offs, mask=offs < n, other=-float('inf'))
    curr_max = tl.max(running_max, axis=0)
    for start in range(BLOCK, n, BLOCK):
        chunk_offs = start + offs
        mask = chunk_offs < n
        chunk = tl.load(x_ptr + chunk_offs, mask=mask, other=-float('inf'))
        chunk_max = tl.max(chunk, axis=0)
        curr_max = tl.where(chunk_max > curr_max, chunk_max, curr_max)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, curr_max)


# 8. Cross-entropy loss (log softmax + gather + negate)
@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr,
                          row_stride, n_classes,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_classes
    logits = tl.load(logits_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    # Log-softmax
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(logits), axis=0))
    log_softmax = logits - log_sum_exp
    # Gather the log prob at the correct label
    label = tl.load(labels_ptr + row)
    # We need log_softmax[label] — use a masked approach
    label_mask = offs == label
    log_prob = tl.sum(tl.where(label_mask, log_softmax, 0.0), axis=0)
    tl.store(loss_ptr + row, -log_prob)


# 9. Pairwise distance matrix (batch of L2 distances between rows)
@triton.jit
def pairwise_sq_dist_kernel(x_ptr, out_ptr, M, D,
                             BLOCK_D: tl.constexpr):
    """Compute squared L2 distance between row i and row j."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D
    xi = tl.load(x_ptr + pid_i * D + offs_d, mask=mask, other=0.0)
    xj = tl.load(x_ptr + pid_j * D + offs_d, mask=mask, other=0.0)
    diff = xi - xj
    dist = tl.sum(diff * diff, axis=0)
    tl.store(out_ptr + pid_i * M + pid_j, dist)


# 10. RoPE (Rotary Position Embedding) - interleaved sin/cos
@triton.jit
def rope_kernel(x_ptr, out_ptr, pos, d_model,
                BLOCK: tl.constexpr):
    """Apply rotary embedding: pairs of dimensions rotated by position-dependent angle."""
    pid = tl.program_id(0)  # sequence position handled by pos arg
    offs = tl.arange(0, BLOCK)
    mask = offs < d_model
    x = tl.load(x_ptr + pid * d_model + offs, mask=mask)
    # Compute frequencies: theta_i = pos / 10000^(2i/d)
    half_d = d_model // 2
    dim_idx = offs % half_d
    freq = pos / tl.exp2(dim_idx.to(tl.float32) * (20.0 / half_d))  # approximate log(10000)*2/d
    cos_val = tl.cos(freq)
    sin_val = tl.sin(freq)
    # Even dims: x * cos - x_paired * sin
    # Odd dims: x * sin + x_paired * cos
    is_even = (offs < half_d)
    # For simplicity: rotate first half with second half
    x_paired = tl.load(x_ptr + pid * d_model + (offs + half_d) % d_model, mask=mask)
    out = tl.where(is_even, x * cos_val - x_paired * sin_val,
                            x * cos_val + x_paired * sin_val)
    tl.store(out_ptr + pid * d_model + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_running_sum():
    chunk_size = 64
    n_chunks = 8
    x = torch.randn(n_chunks * chunk_size, device='mps')
    out = torch.zeros(n_chunks, device='mps')
    running_sum_kernel[(1,)](x, out, n_chunks, chunk_size, BLOCK=64)
    # Compute reference
    ref = torch.zeros(n_chunks, device='mps')
    for i in range(n_chunks):
        ref[i] = x[:chunk_size * (i + 1)].sum()
    err = (out - ref).abs().max().item()
    return err < 1e-2, err


def test_strided_copy():
    n = 256
    # Create a source with stride 3 and destination with stride 2
    src = torch.randn(n * 3, device='mps')
    dst = torch.zeros(n * 2, device='mps')
    strided_copy_kernel[(1,)](src, dst, n, 3, 2, BLOCK=256)
    # Reference: dst[i*2] = src[i*3]
    ref = torch.zeros(n * 2, device='mps')
    for i in range(n):
        ref[i * 2] = src[i * 3]
    err = (dst - ref).abs().max().item()
    return err < 1e-5, err


def test_gelu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gelu_kernel[grid](x, out, n, BLOCK=256)
    # Approximate GELU reference (same formula)
    ref = x * torch.sigmoid(1.702 * x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_softmax_scale_clamp():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    scale = 2.0
    lo = 0.01
    hi = 0.99
    softmax_scale_clamp_kernel[(M,)](x, out, scale, lo, hi, N, N, BLOCK_SIZE=64)
    # Reference
    sm = torch.softmax(x, dim=1) * scale
    ref = sm.clamp(lo, hi)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_gather():
    src_size = 1000
    n = 512
    src = torch.randn(src_size, device='mps')
    idx = torch.randint(0, src_size, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gather_kernel[grid](src, idx, out, n, BLOCK=256)
    ref = src[idx.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_scatter_add():
    n = 512
    n_bins = 64
    src = torch.ones(n, device='mps')
    idx = torch.randint(0, n_bins, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n_bins, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    scatter_add_kernel[grid](src, idx, out, n, BLOCK=256)
    # Reference
    ref = torch.zeros(n_bins, device='mps')
    for i in range(n_bins):
        ref[i] = (idx == i).float().sum()
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_online_max():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    online_max_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x.max()
    err = abs(out.item() - ref.item())
    return err < 1e-5, err


def test_cross_entropy():
    M = 16
    n_classes = 32
    logits = torch.randn(M, n_classes, device='mps')
    labels = torch.randint(0, n_classes, (M,), device='mps', dtype=torch.int32)
    loss = torch.zeros(M, device='mps')
    cross_entropy_kernel[(M,)](logits, labels, loss, n_classes, n_classes, BLOCK_SIZE=32)
    # Reference
    ref = torch.nn.functional.cross_entropy(logits, labels.long(), reduction='none')
    err = (loss - ref).abs().max().item()
    return err < 1e-3, err


def test_pairwise_dist():
    M = 8
    D = 32
    x = torch.randn(M, D, device='mps')
    out = torch.zeros(M, M, device='mps')
    pairwise_sq_dist_kernel[(M, M)](x, out, M, D, BLOCK_D=32)
    # Reference
    ref = torch.cdist(x, x, p=2) ** 2
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_rope():
    seq_len = 4
    d_model = 32
    x = torch.randn(seq_len, d_model, device='mps')
    out = torch.zeros(seq_len, d_model, device='mps')
    pos = 5.0  # position index
    rope_kernel[(seq_len,)](x, out, pos, d_model, BLOCK=32)
    # Just verify output is finite and different from input
    finite = torch.isfinite(out).all().item()
    changed = (out - x).abs().max().item() > 1e-6
    return finite and changed, 0.0 if (finite and changed) else 1.0


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 6) ===\n")

    tests = [
        ("Running Sum (loop accum)", test_running_sum),
        ("Strided Copy", test_strided_copy),
        ("GELU Activation", test_gelu),
        ("Softmax+Scale+Clamp (pipeline)", test_softmax_scale_clamp),
        ("Gather (indirect load)", test_gather),
        ("Scatter Add (indirect atomic)", test_scatter_add),
        ("Online Max (streaming)", test_online_max),
        ("Cross-Entropy Loss", test_cross_entropy),
        ("Pairwise Distance", test_pairwise_dist),
        ("RoPE (rotary embed)", test_rope),
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
