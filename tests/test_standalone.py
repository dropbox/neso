#!/usr/bin/env python3
"""
Standalone end-to-end test for the Neso backend.
Demonstrates the full pipeline: Triton-like kernel definition -> MSL -> Metal execution.
Works without the full Triton build by using the codegen directly.
"""
import sys
import os
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neso.backend.codegen import ttir_to_msl

try:
    import Metal
except ImportError:
    print("PyObjC Metal framework not available. Install with: pip install pyobjc-framework-Metal")
    sys.exit(1)


class MetalKernel:
    """A compiled Metal compute kernel ready for execution."""

    def __init__(self, msl_source: str, kernel_name: str, device=None):
        self.device = device or Metal.MTLCreateSystemDefaultDevice()
        self.name = kernel_name
        self.msl_source = msl_source

        # Compile MSL
        options = Metal.MTLCompileOptions.alloc().init()
        options.setFastMathEnabled_(True)
        library, error = self.device.newLibraryWithSource_options_error_(
            msl_source, options, None
        )
        if error is not None:
            raise RuntimeError(f"MSL compilation failed: {error.localizedDescription()}")

        function = library.newFunctionWithName_(kernel_name)
        if function is None:
            raise RuntimeError(f"Kernel '{kernel_name}' not found in compiled library")

        self.pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if error is not None:
            raise RuntimeError(f"Pipeline creation failed: {error.localizedDescription()}")

        self.command_queue = self.device.newCommandQueue()
        self.max_threads = self.pipeline.maxTotalThreadsPerThreadgroup()

    @classmethod
    def from_ttir(cls, ttir_text: str, block_size: int = 256, device=None):
        """Create a MetalKernel from Triton IR text."""
        msl_source, kernel_name = ttir_to_msl(ttir_text, block_size=block_size)
        return cls(msl_source, kernel_name, device)

    def __call__(self, grid, block_size, *buffers):
        """Execute the kernel with the given grid/block dimensions and buffers."""
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)

        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        threadgroup_size = Metal.MTLSizeMake(block_size, 1, 1)
        if isinstance(grid, int):
            grid_size = Metal.MTLSizeMake(grid, 1, 1)
        else:
            grid_size = Metal.MTLSizeMake(*grid, *([1] * (3 - len(grid))))

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        if command_buffer.error() is not None:
            raise RuntimeError(f"Kernel execution failed: {command_buffer.error().localizedDescription()}")


class MetalArray:
    """Simple wrapper for Metal buffers with numpy-like interface."""

    def __init__(self, data, dtype='f', device=None):
        self.device = device or Metal.MTLCreateSystemDefaultDevice()
        self.dtype = dtype
        self._dtype_size = struct.calcsize(dtype)

        if isinstance(data, (list, tuple, range)):
            self.size = len(data)
            raw = struct.pack(f'{self.size}{dtype}', *data)
            self.buffer = self.device.newBufferWithBytes_length_options_(
                raw, len(raw), Metal.MTLResourceStorageModeShared
            )
        elif isinstance(data, int):
            # Allocate empty buffer
            self.size = data
            self.buffer = self.device.newBufferWithLength_options_(
                data * self._dtype_size, Metal.MTLResourceStorageModeShared
            )
        elif isinstance(data, bytes):
            self.size = len(data) // self._dtype_size
            self.buffer = self.device.newBufferWithBytes_length_options_(
                data, len(data), Metal.MTLResourceStorageModeShared
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @classmethod
    def zeros(cls, size, dtype='f', device=None):
        arr = cls(size, dtype=dtype, device=device)
        return arr

    @classmethod
    def scalar(cls, value, dtype='i', device=None):
        """Create a single-element buffer for a scalar value."""
        dev = device or Metal.MTLCreateSystemDefaultDevice()
        raw = struct.pack(dtype, value)
        arr = cls.__new__(cls)
        arr.device = dev
        arr.dtype = dtype
        arr._dtype_size = struct.calcsize(dtype)
        arr.size = 1
        arr.buffer = dev.newBufferWithBytes_length_options_(
            raw, len(raw), Metal.MTLResourceStorageModeShared
        )
        return arr

    def tolist(self):
        """Read buffer contents back to a Python list."""
        contents = self.buffer.contents()
        raw = contents.as_buffer(self.size * self._dtype_size)
        return list(struct.unpack(f'{self.size}{self.dtype}', raw))

    def __len__(self):
        return self.size

    def __repr__(self):
        data = self.tolist()
        if len(data) > 10:
            data_str = str(data[:5])[:-1] + f', ... ({len(data)} total)]'
        else:
            data_str = str(data)
        return f"MetalArray({data_str})"


# ---- Test kernels ----

def test_vector_add():
    """Test: z = x + y"""
    print("--- Test: Vector Addition (z = x + y) ---")

    ttir = """
    module {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
        %3 = tt.splat %1 : i32 -> tensor<256xi32>
        %4 = arith.addi %3, %2 : tensor<256xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
        %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
        %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
        %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
        %12 = tt.load %11, %6 : tensor<256x!tt.ptr<f32>>
        %13 = arith.addf %9, %12 : tensor<256xf32>
        %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
        %15 = tt.addptr %14, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
        tt.store %15, %13, %6 : tensor<256x!tt.ptr<f32>>
        tt.return
      }
    }
    """

    N = 100000
    BLOCK_SIZE = 256

    kernel = MetalKernel.from_ttir(ttir, block_size=BLOCK_SIZE)

    x = MetalArray([float(i) for i in range(N)])
    y = MetalArray([float(i * 0.5) for i in range(N)])
    z = MetalArray.zeros(N)
    n = MetalArray.scalar(N, dtype='i')

    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Warmup
    kernel(grid, BLOCK_SIZE, x.buffer, y.buffer, z.buffer, n.buffer)

    # Benchmark
    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        kernel(grid, BLOCK_SIZE, x.buffer, y.buffer, z.buffer, n.buffer)
    elapsed = time.perf_counter() - start

    # Verify
    result = z.tolist()
    errors = 0
    for i in range(N):
        expected = float(i) + float(i * 0.5)
        if abs(result[i] - expected) > 1e-3:
            errors += 1
            if errors <= 3:
                print(f"  Mismatch at [{i}]: expected {expected}, got {result[i]}")

    if errors == 0:
        bw = N * 4 * 3 * iters / elapsed / 1e9  # 3 arrays * 4 bytes * N elements
        print(f"  PASSED: {N} elements correct")
        print(f"  Throughput: {bw:.1f} GB/s ({iters} iterations in {elapsed*1000:.1f}ms)")
    else:
        print(f"  FAILED: {errors} mismatches")
    return errors == 0


def test_element_wise_ops():
    """Test: y = exp(x * 2.0 + 1.0)"""
    print("\n--- Test: Element-wise ops (y = exp(x * scale)) ---")

    ttir = """
    module {
      tt.func public @ewise_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: f32, %arg3: i32) {
        %c512_i32 = arith.constant 512 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c512_i32 : i32
        %2 = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
        %3 = tt.splat %1 : i32 -> tensor<512xi32>
        %4 = arith.addi %3, %2 : tensor<512xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
        %9 = tt.load %8, %6 : tensor<512x!tt.ptr<f32>>
        %10 = tt.splat %arg2 : f32 -> tensor<512xf32>
        %11 = arith.mulf %9, %10 : tensor<512xf32>
        %12 = math.exp %11 : tensor<512xf32>
        %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
        %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
        tt.store %14, %12, %6 : tensor<512x!tt.ptr<f32>>
        tt.return
      }
    }
    """

    import math
    N = 1024
    BLOCK_SIZE = 512
    SCALE = 0.01

    kernel = MetalKernel.from_ttir(ttir, block_size=BLOCK_SIZE)

    x = MetalArray([float(i) for i in range(N)])
    y = MetalArray.zeros(N)
    scale = MetalArray.scalar(SCALE, dtype='f')
    n = MetalArray.scalar(N, dtype='i')

    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    kernel(grid, BLOCK_SIZE, x.buffer, y.buffer, scale.buffer, n.buffer)

    result = y.tolist()
    errors = 0
    for i in range(N):
        expected = math.exp(float(i) * SCALE)
        if abs(result[i] - expected) / max(abs(expected), 1e-7) > 1e-4:
            errors += 1
            if errors <= 3:
                print(f"  Mismatch at [{i}]: expected {expected:.6f}, got {result[i]:.6f}")

    if errors == 0:
        print(f"  PASSED: {N} elements correct (exp with scale={SCALE})")
    else:
        print(f"  FAILED: {errors} mismatches")
    return errors == 0


def test_large_scale():
    """Stress test with large arrays."""
    print("\n--- Test: Large Scale (10M elements) ---")

    ttir = """
    module {
      tt.func public @scale_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
        %c1024_i32 = arith.constant 1024 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c1024_i32 : i32
        %2 = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
        %3 = tt.splat %1 : i32 -> tensor<1024xi32>
        %4 = arith.addi %3, %2 : tensor<1024xi32>
        %5 = tt.splat %arg2 : i32 -> tensor<1024xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
        %10 = arith.addf %9, %9 : tensor<1024xf32>
        %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f32>>
        tt.return
      }
    }
    """

    N = 10_000_000
    BLOCK_SIZE = 1024

    kernel = MetalKernel.from_ttir(ttir, block_size=BLOCK_SIZE)

    # Create large buffers - use raw bytes for efficiency
    x_data = struct.pack(f'{N}f', *[float(i % 1000) for i in range(N)])
    x = MetalArray(x_data)
    y = MetalArray.zeros(N)
    n = MetalArray.scalar(N, dtype='i')

    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Warmup
    kernel(grid, BLOCK_SIZE, x.buffer, y.buffer, n.buffer)

    # Benchmark
    start = time.perf_counter()
    iters = 20
    for _ in range(iters):
        kernel(grid, BLOCK_SIZE, x.buffer, y.buffer, n.buffer)
    elapsed = time.perf_counter() - start

    # Spot check a few values
    result = y.tolist()
    ok = True
    for i in [0, 1, 999, 10000, N - 1]:
        expected = float(i % 1000) * 2.0
        if abs(result[i] - expected) > 0.1:
            print(f"  Mismatch at [{i}]: expected {expected}, got {result[i]}")
            ok = False

    if ok:
        bw = N * 4 * 2 * iters / elapsed / 1e9  # 2 arrays * 4 bytes
        print(f"  PASSED: {N:,} elements, spot checks correct")
        print(f"  Throughput: {bw:.1f} GB/s ({iters} iterations in {elapsed*1000:.1f}ms)")
    else:
        print(f"  FAILED: verification errors")
    return ok


if __name__ == "__main__":
    device = Metal.MTLCreateSystemDefaultDevice()
    print(f"Metal Device: {device.name()}")
    print(f"Unified Memory: {device.hasUnifiedMemory()}")
    print(f"Max Threadgroup Memory: {device.maxThreadgroupMemoryLength()} bytes")
    print()

    results = []
    results.append(("Vector Addition", test_vector_add()))
    results.append(("Element-wise Ops", test_element_wise_ops()))
    results.append(("Large Scale", test_large_scale()))

    print(f"\n{'='*50}")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*50}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
