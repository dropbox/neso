#!/usr/bin/env python3
"""Test real @triton.jit kernels through the full compilation pipeline.

Tests: Triton Python AST -> TTIR -> MSL -> Metal GPU -> correct results
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)


@triton.jit
def saxpy_kernel(x_ptr, y_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, alpha * x, mask=mask)


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x > 0, x, 0.0), mask=mask)


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * x, mask=mask)


@triton.jit
def fused_add_mul_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
    """out = (a + b) * c"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, (a + b) * c, mask=mask)


@triton.jit
def clamp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.minimum(tl.maximum(x, -1.0), 1.0), mask=mask)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def atomic_add_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.atomic_add(out_ptr + offsets, x, mask=mask)


@triton.jit
def grid_stride_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Uses tl.num_programs for grid-stride loop."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for i in range(pid, tl.cdiv(n, BLOCK), num_pids):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def cumsum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def dropout_kernel(x_ptr, out_ptr, seed, p, n, BLOCK: tl.constexpr):
    """Dropout with Philox PRNG (tl.rand, tt.mulhiui)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    rand = tl.rand(seed, offs)
    keep = rand > p
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """LogSumExp with conditional store (scf.if)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    result = max_val + tl.log(sum_exp)
    if pid == 0:
        tl.store(out_ptr, result)


@triton.jit
def layernorm_kernel(x_ptr, out_ptr, weight_ptr, bias_ptr, row_stride, n_cols, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xc * rstd
    w = tl.load(weight_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    out = xn * w + b
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def cross_entropy_kernel(logits_ptr, targets_ptr, loss_ptr,
                         row_stride, n_cols,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    logits = tl.load(logits_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(logits), axis=0))
    log_softmax = logits - log_sum_exp
    target = tl.load(targets_ptr + row)
    target_log_prob = tl.sum(tl.where(offs == target, log_softmax, 0.0), axis=0)
    tl.store(loss_ptr + row, -target_log_prob)


@triton.jit
def int_ops_kernel(x_ptr, out_shl_ptr, out_shr_ptr, out_and_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_shl_ptr + offs, x << 2, mask=mask)
    tl.store(out_shr_ptr + offs, x >> 1, mask=mask)
    tl.store(out_and_ptr + offs, x & 0xFF, mask=mask)


@triton.jit
def matmul_kernel(
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


# ============================================================
# Test runners
# ============================================================

def test_add():
    n = 2048
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_mul():
    n = 2048
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    ref = x * y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_saxpy():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    alpha = 3.14
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    saxpy_kernel[grid](x, out, alpha, n, BLOCK_SIZE=256)
    ref = alpha * x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_relu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=256)
    ref = torch.relu(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_softmax():
    M, N = 16, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N, BLOCK_SIZE=128)
    ref = torch.softmax(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_softmax_256():
    """Softmax with larger BLOCK_SIZE (multi-SIMD-group reduce)."""
    M, N = 8, 256
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N, BLOCK_SIZE=256)
    ref = torch.softmax(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_square():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    square_kernel[grid](x, out, n, BLOCK_SIZE=256)
    ref = x * x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_fused_add_mul():
    n = 2048
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    c = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_add_mul_kernel[grid](a, b, c, out, n, BLOCK_SIZE=256)
    ref = (a + b) * c
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_clamp():
    n = 2048
    x = torch.randn(n, device='mps') * 3
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    clamp_kernel[grid](x, out, n, BLOCK_SIZE=256)
    ref = torch.clamp(x, -1, 1)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_gelu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=256)
    ref = torch.nn.functional.gelu(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_atomic_add():
    n = 2048
    x = torch.ones(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    atomic_add_kernel[grid](x, out, n, BLOCK_SIZE=256)
    ref = x.clone()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_grid_stride():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid_stride_kernel[(4,)](x, out, n, BLOCK=256)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_cumsum():
    n = 256
    x = torch.ones(n, device='mps')
    out = torch.zeros(n, device='mps')
    cumsum_kernel[(1,)](x, out, n, BLOCK=256)
    ref = torch.arange(1, n + 1, device='mps', dtype=torch.float32)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def _run_matmul(M, N, K, BM, BN, BK):
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    ref = a @ b
    err = (c - ref).abs().max().item()
    # Matmul accumulates K products so use a relative tolerance
    tol = 1e-3 * (K ** 0.5)
    return err < tol, err


def test_dropout():
    n = 4096
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    dropout_kernel[grid](x, out, 42, 0.5, n, BLOCK=256)
    ratio = (out == 0).sum().item() / n
    nonzero = out != 0
    scale_err = (out[nonzero] / x[nonzero] - 2.0).abs().max().item() if nonzero.sum() > 0 else float('inf')
    ok = 0.3 < ratio < 0.7 and scale_err < 1e-5
    return ok, scale_err


def test_logsumexp():
    n = 256
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    logsumexp_kernel[(1,)](x, out, n, BLOCK_SIZE=256)
    ref = torch.logsumexp(x, dim=0)
    err = abs(out.item() - ref.item())
    return err < 1e-4, err


def test_layernorm():
    M, N = 16, 128
    x = torch.randn(M, N, device='mps')
    w = torch.ones(N, device='mps')
    b = torch.zeros(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    layernorm_kernel[(M,)](x, out, w, b, N, N, 1e-5, BLOCK_SIZE=128)
    ref = torch.nn.functional.layer_norm(x, [N], w, b)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_silu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    silu_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.nn.functional.silu(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_cross_entropy():
    M, N = 32, 64
    logits = torch.randn(M, N, device='mps')
    targets = torch.randint(0, N, (M,), device='mps', dtype=torch.int32)
    loss = torch.zeros(M, device='mps')
    cross_entropy_kernel[(M,)](logits, targets, loss, N, N, BLOCK_SIZE=64)
    ref = torch.nn.functional.cross_entropy(logits, targets.long(), reduction='none')
    err = (loss - ref).abs().max().item()
    return err < 1e-4, err


def test_int_ops():
    n = 2048
    x = torch.randint(0, 1024, (n,), device='mps', dtype=torch.int32)
    out_shl = torch.zeros(n, device='mps', dtype=torch.int32)
    out_shr = torch.zeros(n, device='mps', dtype=torch.int32)
    out_and = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    int_ops_kernel[grid](x, out_shl, out_shr, out_and, n, BLOCK=256)
    err = max(
        (out_shl - (x << 2)).abs().max().item(),
        (out_shr - (x >> 1)).abs().max().item(),
        (out_and - (x & 0xFF)).abs().max().item(),
    )
    return err == 0, err


def test_matmul_32():
    return _run_matmul(32, 32, 32, 32, 32, 32)


def test_matmul_64():
    return _run_matmul(64, 64, 64, 32, 32, 32)


def test_matmul_128():
    return _run_matmul(128, 128, 128, 32, 32, 32)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== @triton.jit Metal Backend Tests ===\n")

    tests = [
        ("Vector Add", test_add),
        ("Vector Mul", test_mul),
        ("SAXPY (scalar*x)", test_saxpy),
        ("ReLU (tl.where)", test_relu),
        ("Square (x*x)", test_square),
        ("Fused Add+Mul", test_fused_add_mul),
        ("Softmax (128)", test_softmax),
        ("Softmax (256)", test_softmax_256),
        ("Clamp (maxnum/minnum)", test_clamp),
        ("GELU (erf)", test_gelu),
        ("Atomic Add", test_atomic_add),
        ("Grid Stride (num_programs)", test_grid_stride),
        ("Cumsum (scan)", test_cumsum),
        ("Dropout (tl.rand)", test_dropout),
        ("LogSumExp (scf.if)", test_logsumexp),
        ("LayerNorm", test_layernorm),
        ("SiLU (x*sigmoid)", test_silu),
        ("Cross Entropy", test_cross_entropy),
        ("Int Ops (shl/shr/and)", test_int_ops),
        ("Matmul 32x32", test_matmul_32),
        ("Matmul 64x64", test_matmul_64),
        ("Matmul 128x128", test_matmul_128),
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
