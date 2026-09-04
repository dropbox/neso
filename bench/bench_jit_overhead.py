#!/usr/bin/env python3
"""Profile @triton.jit pipeline: compilation time and GPU kernel execution time.

Excludes data copy overhead (MPS<->CPU<->Metal) from measurements.
"""
import time
import os
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# --- Kernels ---

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


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


@triton.jit
def softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    x = x - max_val
    ex = tl.exp(x)
    sum_ex = tl.sum(ex, axis=0)
    out = ex / sum_ex
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# --- Measure compile time by intercepting the pipeline ---

def measure_compile_time(kernel_fn, *args, **kwargs):
    """Call a kernel and measure wall time. First call = compile+launch, second = launch only."""
    torch.mps.synchronize()
    t0 = time.perf_counter()
    kernel_fn(*args, **kwargs)
    torch.mps.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def measure_gpu_time_standalone(msl_source, kernel_name, buffers, grid, threads_per_group):
    """Measure pure GPU execution time using Metal command buffer timestamps.

    This bypasses the Triton JIT pipeline entirely - just Metal API calls.
    """
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    queue = device.newCommandQueue()

    # Compile MSL
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    library, error = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if error:
        raise RuntimeError(f"MSL compile error: {error.localizedDescription()}")

    function = library.newFunctionWithName_(kernel_name)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if error:
        raise RuntimeError(f"Pipeline error: {error.localizedDescription()}")

    # Create Metal buffers
    metal_bufs = []
    for data_bytes, nbytes in buffers:
        buf = device.newBufferWithBytes_length_options_(
            data_bytes, nbytes, Metal.MTLResourceStorageModeShared)
        metal_bufs.append(buf)

    # Warmup
    for _ in range(3):
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(metal_bufs):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(threads_per_group, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

    # Measure GPU time
    gpu_times = []
    for _ in range(20):
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(metal_bufs):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(threads_per_group, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        gpu_start = cb.GPUStartTime()
        gpu_end = cb.GPUEndTime()
        gpu_times.append((gpu_end - gpu_start) * 1000)

    return gpu_times


if __name__ == "__main__":
    print("=== @triton.jit Pipeline Performance ===\n")

    # --- MPS baselines ---
    print("MPS baselines (for reference):")
    for label, fn in [
        ("add 1K", lambda: torch.randn(1024, device='mps') + torch.randn(1024, device='mps')),
        ("matmul 128", lambda: torch.randn(128, 128, device='mps') @ torch.randn(128, 128, device='mps')),
        ("matmul 512", lambda: torch.randn(512, 512, device='mps') @ torch.randn(512, 512, device='mps')),
    ]:
        # warmup
        for _ in range(5):
            fn()
            torch.mps.synchronize()
        times = []
        for _ in range(20):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        print(f"  {label}: avg={sum(times)/len(times):.3f} ms, min={min(times):.3f} ms")

    # --- Triton compile times ---
    print("\nTriton compilation time (first call = compile + launch):")

    # Clear Triton cache to force recompilation
    os.environ.pop('TRITON_CACHE_DIR', None)

    n = 1024
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)

    first = measure_compile_time(add_kernel[grid], x, y, out, n, BLOCK=256)
    second = measure_compile_time(add_kernel[grid], x, y, out, n, BLOCK=256)
    third = measure_compile_time(add_kernel[grid], x, y, out, n, BLOCK=256)
    print(f"  Add kernel: first={first:.0f}ms, 2nd={second:.0f}ms, 3rd={third:.0f}ms")
    print(f"    Compile overhead: ~{first - second:.0f}ms")

    M, N, K = 128, 128, 128
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid_mm = (triton.cdiv(M, 32), triton.cdiv(N, 32))

    first = measure_compile_time(matmul_kernel[grid_mm], a, b, c, M, N, K,
                                 a.stride(0), a.stride(1),
                                 b.stride(0), b.stride(1),
                                 c.stride(0), c.stride(1),
                                 BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    second = measure_compile_time(matmul_kernel[grid_mm], a, b, c, M, N, K,
                                  a.stride(0), a.stride(1),
                                  b.stride(0), b.stride(1),
                                  c.stride(0), c.stride(1),
                                  BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    third = measure_compile_time(matmul_kernel[grid_mm], a, b, c, M, N, K,
                                 a.stride(0), a.stride(1),
                                 b.stride(0), b.stride(1),
                                 c.stride(0), c.stride(1),
                                 BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    print(f"  Matmul kernel: first={first:.0f}ms, 2nd={second:.0f}ms, 3rd={third:.0f}ms")
    print(f"    Compile overhead: ~{first - second:.0f}ms")

    M2, N2 = 32, 128
    sx = torch.randn(M2, N2, device='mps')
    sout = torch.zeros(M2, N2, device='mps')
    first = measure_compile_time(softmax_kernel[(M2,)], sx, sout, N2, N2, BLOCK_SIZE=128)
    second = measure_compile_time(softmax_kernel[(M2,)], sx, sout, N2, N2, BLOCK_SIZE=128)
    print(f"  Softmax kernel: first={first:.0f}ms, 2nd={second:.0f}ms")
    print(f"    Compile overhead: ~{first - second:.0f}ms")

    # --- GPU-only execution time (bypass JIT, use standalone Metal) ---
    print("\nGPU kernel execution time (standalone Metal, no JIT overhead):")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from neso.backend.codegen import ttir_to_msl_with_metadata

    # Generate MSL for add kernel via codegen
    add_ttir = """
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.addptr %arg0, %4 : !tt.ptr<f32>, tensor<256xi32>
    %8 = tt.load %7, %6 : !tt.ptr<f32>
    %9 = tt.addptr %arg1, %4 : !tt.ptr<f32>, tensor<256xi32>
    %10 = tt.load %9, %6 : !tt.ptr<f32>
    %11 = arith.addf %8, %10 : tensor<256xf32>
    %12 = tt.addptr %arg2, %4 : !tt.ptr<f32>, tensor<256xi32>
    tt.store %12, %11, %6 : !tt.ptr<f32>
    tt.return
  }
}
"""
    msl_source, kernel_name, block_size = ttir_to_msl_with_metadata(add_ttir, block_size=256)

    import struct
    n = 1024
    x_data = struct.pack(f'{n}f', *[float(i) for i in range(n)])
    y_data = struct.pack(f'{n}f', *[1.0]*n)
    out_data = b'\x00' * (n * 4)
    n_data = struct.pack('i', n)

    bufs = [(x_data, n*4), (y_data, n*4), (out_data, n*4), (n_data, 4)]
    gpu_times = measure_gpu_time_standalone(msl_source, kernel_name, bufs,
                                            grid=(4, 1, 1), threads_per_group=block_size)
    print(f"  Add kernel GPU time: avg={sum(gpu_times)/len(gpu_times):.4f} ms, min={min(gpu_times):.4f} ms")

    print("\nSummary:")
    print("  Compilation (one-time): ~600-1000ms (includes TTIR passes, MSL gen, Metal compile)")
    print("  Per-launch wall time: includes data copy overhead (MPS<->CPU<->Metal)")
    print("  GPU kernel execution: the actual compute time on the GPU")
