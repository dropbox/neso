"""
Standalone test for the Metal MSL code generator.
Tests TTIR (MLIR text) -> MSL translation without requiring the full Triton build.
"""
import os
import sys

import pytest

# Add the backend to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neso.backend.codegen import ttir_to_hlsl, ttir_to_hlsl_with_metadata, ttir_to_msl
from neso.backend.codegen.analysis import build_op_map, compute_liveness
from neso.backend.codegen.hlsl_emitter import HLSLEmitter
from neso.backend.codegen.ir import Op, TType
from neso.backend.codegen.lowering import TritonLowering
from neso.backend.codegen.mlir_walker import walk_module_from_text
from neso.backend.codegen.model import UnsupportedOperationError
from neso.backend.codegen.msl_emitter import MSLEmitter

# ---------------------------------------------------------------------------
# Test MLIR inputs (representative TTIR for common Triton kernels)
# ---------------------------------------------------------------------------

# 1. Vector Addition kernel
ADD_KERNEL_TTIR = """
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
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

# 2. Scalar multiply kernel (SAXPY-like)
SAXPY_KERNEL_TTIR = """
module {
  tt.func public @saxpy_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: f32, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg2 : f32 -> tensor<1024xf32>
    %11 = arith.mulf %9, %10 : tensor<1024xf32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %13, %11, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""

# 3. Softmax-like kernel with exp and reduce
SOFTMAX_TTIR = """
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>>
    %10 = math.exp %9 : tensor<128xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %12, %10, %6 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_parser():
    """Test MLIR text parsing."""
    name, args, ops = walk_module_from_text(ADD_KERNEL_TTIR)

    assert name == "add_kernel", f"Expected 'add_kernel', got '{name}'"
    assert len(args) == 4, f"Expected 4 args, got {len(args)}"
    assert args[0].ttype.is_ptr, "First arg should be a pointer"
    assert args[0].ttype.dtype == "f32", f"Expected f32, got {args[0].ttype.dtype}"
    assert args[3].ttype.dtype == "i32", f"Expected i32 scalar, got {args[3].ttype.dtype}"
    assert len(ops) > 0, "Expected operations"

    # Check we found key operations
    op_names = [op.opname for op in ops]
    assert 'tt.get_program_id' in op_names, f"Missing tt.get_program_id in {op_names}"
    assert 'tt.make_range' in op_names, f"Missing tt.make_range in {op_names}"
    assert 'tt.load' in op_names, f"Missing tt.load in {op_names}"
    assert 'arith.addf' in op_names, f"Missing arith.addf in {op_names}"
    assert 'tt.store' in op_names, f"Missing tt.store in {op_names}"
    print("  Parser test passed!")


def test_cast_barrier_deferred_until_intervening_tile_load():
    """Independent cast/load writes should share the load's barrier."""
    cast = Op(
        results=['%p16'], opname='arith.truncf', operands=['%p32'], attrs={},
        type_str='', result_types=[TType(dtype='f16', shape=[16, 32])],
    )
    load = Op(
        results=['%v'], opname='tt.load', operands=['%v_ptr'], attrs={},
        type_str='', result_types=[TType(dtype='f16', shape=[32, 64])],
    )
    dot = Op(
        results=['%o'], opname='tt.dot', operands=['%p16', '%v', '%acc'],
        attrs={}, type_str='',
        result_types=[TType(dtype='f32', shape=[16, 64])],
    )

    deferred = TritonLowering._find_deferred_cast_barriers([cast, load, dot])
    assert deferred == {'%p16'}
    assert TritonLowering._find_deferred_cast_barriers([cast, dot, load]) == set()


def test_add_kernel():
    """Test MSL generation for vector addition."""
    msl, name = ttir_to_msl(ADD_KERNEL_TTIR)

    assert name == "add_kernel"
    assert "kernel void add_kernel" in msl
    assert "device float*" in msl
    assert "buffer(0)" in msl
    assert "buffer(1)" in msl
    assert "buffer(2)" in msl
    assert "buffer(3)" in msl
    assert "_tgid" in msl
    assert "_tid_in_tg" in msl

    print("  Add kernel MSL generation passed!")
    print("--- Generated MSL ---")
    print(msl)
    print("---")


def test_saxpy_kernel():
    """Test MSL generation for SAXPY kernel."""
    msl, name = ttir_to_msl(SAXPY_KERNEL_TTIR)

    assert name == "saxpy_kernel"
    assert "kernel void saxpy_kernel" in msl
    assert "constant float& arg2" in msl  # scalar alpha parameter

    print("  SAXPY kernel MSL generation passed!")
    print("--- Generated MSL ---")
    print(msl)
    print("---")


def test_softmax_kernel():
    """Test MSL generation for softmax kernel."""
    msl, name = ttir_to_msl(SOFTMAX_TTIR)

    assert name == "softmax_kernel"
    assert "exp(" in msl  # Metal's exp function

    print("  Softmax kernel MSL generation passed!")
    print("--- Generated MSL ---")
    print(msl)
    print("---")


def test_unknown_operation_fails():
    op = Op([], "test.unsupported", [], {}, "", [], raw_text="synthetic test op")
    with pytest.raises(UnsupportedOperationError, match="test.unsupported"):
        TritonLowering(MSLEmitter())._gen_op(op)


def test_hex_float_constant_uses_result_type():
    ttir = """
module {
  tt.func public @constant_kernel() {
    %0 = arith.constant 0x3C00 : f16
    tt.return
  }
}
"""
    msl, _ = ttir_to_msl(ttir)
    assert "half" in msl


def test_hlsl_generation_still_uses_shared_lowering():
    hlsl, name = ttir_to_hlsl(ADD_KERNEL_TTIR)
    assert name == "add_kernel"
    assert "RWStructuredBuffer<float> arg0 : register(u0);" in hlsl
    assert "RWStructuredBuffer<float> arg2 : register(u2);" in hlsl
    assert "cbuffer Params : register(b0)" in hlsl
    assert "SM 6.6" in hlsl
    assert "[WaveSize(" not in hlsl
    assert "[numthreads(256, 1, 1)]" in hlsl
    assert "void add_kernel(" in hlsl
    assert "arg0[_idx7]" in hlsl
    assert "arg2[_idx12] = _v11" in hlsl
    assert HLSLEmitter.NUMTHREADS_PLACEHOLDER not in hlsl

    metadata = ttir_to_hlsl_with_metadata(ADD_KERNEL_TTIR, max_threads=64)
    metadata_hlsl, metadata_name, block_size, threads, half4_args = metadata
    assert metadata_name == name
    assert block_size == 256
    assert threads == 64
    assert "[numthreads(64, 1, 1)]" in metadata_hlsl
    assert half4_args == set()

    fixed_hlsl, _ = ttir_to_hlsl(ADD_KERNEL_TTIR, wave_size=32)
    assert "[WaveSize(32)]" in fixed_hlsl

    with pytest.raises(ValueError, match="wave_size must be one of"):
        ttir_to_hlsl(ADD_KERNEL_TTIR, wave_size=24)


@pytest.mark.parametrize("kind", ["si", "ui"])
def test_integer_divrem_reuses_quotient(kind):
    ttir = f"""
module {{
  tt.func public @divrem_kernel(%arg0: i32, %arg1: i32, %arg2: !tt.ptr<i32>) {{
    %rem = arith.rem{kind} %arg0, %arg1 : i32
    %quot = arith.div{kind} %arg0, %arg1 : i32
    %sum = arith.addi %rem, %quot : i32
    tt.store %arg2, %sum : !tt.ptr<i32>
    tt.return
  }}
}}
"""
    hlsl, _ = ttir_to_hlsl(ttir)
    assert hlsl.count(" / ") == 2  # erf helper plus one integer quotient
    assert " % " not in hlsl
    assert "_rem" in hlsl
    assert "_quot" in hlsl


def test_nested_liveness_keeps_outer_values_alive():
    inner = Op(["%inner"], "arith.addi", ["%outer"], {}, "", [])
    loop = Op(["%loop"], "scf.for", ["%start", "%end", "%step"], {}, "", [], body_ops=[inner])
    op_map = build_op_map([loop])
    liveness = compute_liveness([loop])
    assert op_map["%inner"] is inner
    assert liveness["%outer"] > liveness["%start"]


def test_metal_compilation():
    """Test runtime Metal compilation of generated MSL (requires macOS with Metal)."""
    Metal = pytest.importorskip("Metal")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")

    msl, name = ttir_to_msl(ADD_KERNEL_TTIR)

    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)

    library, error = device.newLibraryWithSource_options_error_(msl, options, None)
    assert error is None, f"{error.localizedDescription()}\n{msl}"

    function = library.newFunctionWithName_(name)
    assert function is not None, f"could not find function {name!r}"

    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert error is None, error.localizedDescription()

    print("  Metal compilation test passed!")
    print(f"    Device: {device.name()}")
    print(f"    Max threads/threadgroup: {pipeline.maxTotalThreadsPerThreadgroup()}")


def test_metal_execution():
    """Test actual Metal kernel execution for vector addition."""
    Metal = pytest.importorskip("Metal")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")

    # Generate and compile MSL
    msl, name = ttir_to_msl(ADD_KERNEL_TTIR)

    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    library, error = device.newLibraryWithSource_options_error_(msl, options, None)
    assert error is None, error.localizedDescription()

    function = library.newFunctionWithName_(name)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert error is None, error.localizedDescription()

    # Create test data
    import struct

    N = 512
    block_size = 256

    # Pack data as bytes for Metal buffers
    x_bytes = struct.pack(f'{N}f', *[float(i) for i in range(N)])
    y_bytes = struct.pack(f'{N}f', *[float(i * 2) for i in range(N)])
    out_bytes = struct.pack(f'{N}f', *[0.0] * N)
    n_bytes = struct.pack('i', N)

    # Create Metal buffers
    x_buf = device.newBufferWithBytes_length_options_(
        x_bytes, len(x_bytes), Metal.MTLResourceStorageModeShared
    )
    y_buf = device.newBufferWithBytes_length_options_(
        y_bytes, len(y_bytes), Metal.MTLResourceStorageModeShared
    )
    out_buf = device.newBufferWithLength_options_(
        len(out_bytes), Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithBytes_length_options_(
        n_bytes, len(n_bytes), Metal.MTLResourceStorageModeShared
    )

    # Create command queue and buffer
    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    # Set pipeline and buffers
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
    encoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
    encoder.setBuffer_offset_atIndex_(out_buf, 0, 2)
    encoder.setBuffer_offset_atIndex_(n_buf, 0, 3)

    # Dispatch
    grid_size = (N + block_size - 1) // block_size
    threadgroup_size = Metal.MTLSizeMake(block_size, 1, 1)
    grid = Metal.MTLSizeMake(grid_size, 1, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid, threadgroup_size)

    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    assert command_buffer.error() is None, command_buffer.error().localizedDescription()

    # Read back results using the buffer contents pointer
    result_ptr = out_buf.contents()
    # PyObjC returns an objc.varlist - use as_buffer() to get a memoryview
    result_buf = result_ptr.as_buffer(N * 4)
    result = struct.unpack(f'{N}f', result_buf)

    # Verify
    errors = 0
    for i in range(N):
        expected = float(i) + float(i * 2)
        if abs(result[i] - expected) > 1e-5:
            if errors < 5:
                print(f"  Mismatch at [{i}]: expected {expected}, got {result[i]}")
            errors += 1

    assert errors == 0, f"{errors} mismatches out of {N}"

    print(f"  Metal execution test passed! All {N} elements correct.")
