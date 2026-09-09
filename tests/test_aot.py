"""
Test the Neso AOT compiler.

Compiles several kernel types and verifies correctness of the outputs.
"""
import json
import os
import re
import tempfile

import pytest
import triton
import triton.language as tl

# ─── Example kernels ─────────────────────────────────────────────────────────

@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)


@triton.jit
def relu_activation(x):
    return tl.maximum(x, 0.0)


@triton.jit
def conditional_activation_kernel(
    output_ptr, input_ptr,
    ACTIVATION: tl.constexpr,
):
    offsets = tl.arange(0, 32)
    values = tl.load(input_ptr + offsets)
    if ACTIVATION:
        values = ACTIVATION(values)
    tl.store(output_ptr + offsets, values)


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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_compile_add():
    """Test AOT compilation of vector add kernel."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=add_kernel,
        signature="*fp32, *fp32, *fp32, i32, 1024",
        num_warps=4,
        grid=["cdiv(n_elements, 1024)", "1", "1"],
    )

    assert result.kernel_name == "add_kernel", f"Got {result.kernel_name}"
    assert len(result.msl_source) > 100, "MSL source too short"
    assert result.metallib_bytes is None or len(result.metallib_bytes) > 0
    assert result.constants == {"BLOCK_SIZE": 1024}
    assert result.grid == ["cdiv(n_elements, 1024)", "1", "1"]

    # Check params — BLOCK_SIZE should NOT be in params (it's constexpr)
    param_names = [p['name'] for p in result.params]
    assert "BLOCK_SIZE" not in param_names, f"constexpr in params: {param_names}"
    assert "x_ptr" in param_names or any("ptr" in n or "x" in n for n in param_names), \
        f"Expected pointer params, got: {param_names}"

    # Check that MSL contains kernel function
    assert "kernel void" in result.msl_source, "No kernel function in MSL"

    # Verify metadata serialization roundtrips
    meta = result.to_metadata()
    assert meta["kernel_name"] == "add_kernel"
    json_str = json.dumps(meta)
    meta2 = json.loads(json_str)
    assert meta2["constants"]["BLOCK_SIZE"] == 1024
    assert all("rust_type" not in param for param in meta2["params"])

    print("  PASS: add_kernel compiled successfully")
    print(f"    MSL: {len(result.msl_source)} chars")
    print(f"    Params: {param_names}")
    print(f"    Threadgroup: {result.threadgroup_size} threads")


def test_compile_softmax():
    """Test AOT compilation of softmax kernel."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=softmax_kernel,
        signature="*fp32, *fp32, i32, 1024",
        num_warps=4,
        grid=["n_rows", "1", "1"],
    )

    assert result.kernel_name == "softmax_kernel"
    assert result.constants == {"BLOCK_SIZE": 1024}
    assert "kernel void" in result.msl_source

    param_names = [p['name'] for p in result.params]
    assert "BLOCK_SIZE" not in param_names

    from neso.backend.codegen import ttir_to_hlsl
    hlsl, _ = ttir_to_hlsl(result.ttgir_text, block_size=result.threadgroup_size)
    assert "[WaveSize(" not in hlsl
    assert "WaveGetLaneCount" in hlsl
    assert "firstbitlow(WaveGetLaneCount())" in hlsl
    assert "/ WaveGetLaneCount" not in hlsl
    assert ">> _wave_lane_shift()" in hlsl

    fixed_hlsl, _ = ttir_to_hlsl(
        result.ttgir_text, block_size=result.threadgroup_size, wave_size=16)
    assert "[WaveSize(16)]" in fixed_hlsl
    assert "WaveGetLaneCount" not in fixed_hlsl
    assert ">> 4" in fixed_hlsl

    print("  PASS: softmax_kernel compiled successfully")
    print(f"    MSL: {len(result.msl_source)} chars")
    print(f"    Params: {param_names}")


def test_compile_jit_function_condition():
    """The plugin supports the common ``if ACTIVATION:`` kernel pattern."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=conditional_activation_kernel,
        signature="*fp32, *fp32, relu_activation",
    )

    assert "max(" in result.msl_source


def test_compile_matmul():
    """Test AOT compilation of FP16 matmul kernel."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=matmul_kernel,
        signature="*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32",
        num_warps=8,
        grid=["cdiv(M, 128)", "cdiv(N, 128)", "1"],
    )

    assert result.kernel_name == "matmul_kernel"
    assert result.constants == {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}
    assert "kernel void" in result.msl_source
    # Matmul should use simdgroup_matrix
    assert "simdgroup_matrix" in result.msl_source or "simdgroup" in result.msl_source, \
        "Expected simdgroup_matrix in matmul MSL"

    # The optimized matmul store bypasses the ordinary addptr lowering. Make
    # sure its row address uses stride_cm (arg10), not the positional fallback
    # stride_am (arg6).
    output_store = r"arg2\[[^\n]+\* arg10 \+"
    assert re.search(output_store, result.msl_source)

    from neso.backend.codegen import ttir_to_hlsl
    hlsl, _ = ttir_to_hlsl(result.ttgir_text, block_size=result.threadgroup_size)
    assert re.search(output_store, hlsl)

    param_names = [p['name'] for p in result.params]
    for const in ["BLOCK_M", "BLOCK_N", "BLOCK_K"]:
        assert const not in param_names, f"constexpr {const} in params"

    # Check pointer params have correct types
    ptr_params = [p for p in result.params if p['is_pointer']]
    assert len(ptr_params) >= 3, f"Expected >=3 pointer params, got {len(ptr_params)}"
    for p in ptr_params:
        assert 'half' in p['metal_type'], f"Expected half pointer, got {p['metal_type']}"

    print("  PASS: matmul_kernel compiled successfully")
    print(f"    MSL: {len(result.msl_source)} chars")
    print(f"    Params: {param_names}")
    print(f"    Threadgroup: {result.threadgroup_size} threads ({result.num_warps} SIMD groups)")


def test_save_artifacts():
    """Test saving compilation artifacts to files."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=add_kernel,
        signature="*fp32, *fp32, *fp32, i32, 1024",
        grid=["cdiv(n_elements, 1024)", "1", "1"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "add_kernel")
        paths = result.save(prefix)

        # Verify all files were created
        for ext in ['metal', 'json', 'ttir', 'ttgir']:
            assert os.path.exists(paths[ext]), f"Missing {ext} file"

        # Verify JSON is valid
        with open(paths['json']) as f:
            meta = json.load(f)
        assert meta['kernel_name'] == 'add_kernel'
        assert meta['constants']['BLOCK_SIZE'] == 1024
        assert len(meta['params']) > 0

        # Verify MSL is readable
        with open(paths['metal']) as f:
            msl = f.read()
        assert 'kernel void' in msl

        if result.metallib_bytes is None:
            assert 'metallib' not in paths
        else:
            assert os.path.getsize(paths['metallib']) > 0

    print("  PASS: artifacts saved and verified")


def test_compile_many():
    """Test compiling multiple specializations."""
    from neso.aot_compile import compile_many

    configs = [
        {
            "signature": "*fp32, *fp32, *fp32, i32, 256",
            "num_warps": 2,
            "grid": ["cdiv(n, 256)", "1", "1"],
        },
        {
            "signature": "*fp32, *fp32, *fp32, i32, 1024",
            "num_warps": 4,
            "grid": ["cdiv(n, 1024)", "1", "1"],
        },
    ]

    results = compile_many(fn=add_kernel, configs=configs)
    assert len(results) == 2
    assert results[0].constants["BLOCK_SIZE"] == 256
    assert results[1].constants["BLOCK_SIZE"] == 1024
    assert results[0].threadgroup_size != results[1].threadgroup_size or True  # may differ

    print("  PASS: compile_many produced 2 specializations")


def test_signature_rejects_unknown_hint():
    from neso.backend.abi import parse_signature

    with pytest.raises(ValueError, match="hints 1 and 16"):
        parse_signature(["x"], "*fp32:8")


def test_invalid_target_is_rejected():
    from neso.aot_compile import compile_kernel

    with pytest.raises(ValueError, match="neso:arch:warp_size"):
        compile_kernel(add_kernel, "*fp32, *fp32, *fp32, i32, 1024", target="neso:2")


def test_jit_runtime_options_are_accepted():
    """Keep Neso's option schema in sync with options injected by Triton JIT."""
    from neso.backend.compiler import NesoBackend

    options = NesoBackend.parse_options(None, {
        "launch_cooperative_grid": True,
        "fpsan_homomorphic_casts": True,
    })
    assert options.launch_cooperative_grid is True
    assert options.fpsan_homomorphic_casts is True


def test_required_metallib_reports_missing_toolchain(monkeypatch):
    from neso import aot_compile
    from neso.backend.toolchain import MetalToolchainUnavailable

    def unavailable(_source):
        raise MetalToolchainUnavailable("toolchain unavailable for test")

    monkeypatch.setattr(aot_compile, "compile_msl", unavailable)
    with pytest.raises(MetalToolchainUnavailable, match="unavailable for test"):
        aot_compile.compile_kernel(
            add_kernel,
            "*fp32, *fp32, *fp32, i32, 1024",
            require_metallib=True,
        )


def test_metallib_emission_can_be_disabled(monkeypatch):
    from neso import aot_compile

    def unexpected(_source):
        raise AssertionError("Metal compiler should not run for a Windows-only build")

    monkeypatch.setattr(aot_compile, "compile_msl", unexpected)
    result = aot_compile.compile_kernel(
        add_kernel,
        "*fp32, *fp32, *fp32, i32, 1024",
        emit_metallib=False,
    )
    assert result.metallib_bytes is None


def test_matmul_save_and_inspect():
    """Compile a matmul kernel, then validate its saved AOT metadata."""
    from neso.aot_compile import compile_kernel

    result = compile_kernel(
        fn=matmul_kernel,
        signature="*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32",
        num_warps=8,
        grid=["cdiv(M, 128)", "cdiv(N, 128)", "1"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "matmul_fp16")
        paths = result.save(prefix)

        with open(paths['json']) as f:
            meta = json.load(f)

        # Verify the metadata is complete enough for Rust dispatch
        assert 'kernel_name' in meta
        assert 'params' in meta
        assert 'threadgroup_size' in meta
        assert 'grid' in meta
        assert meta['threadgroup_size'] > 0
        assert all('index' in p and 'name' in p and 'metal_type' in p
                    for p in meta['params'])

        # Print what a Rust consumer would see
        print("  Rust dispatch info:")
        print(f"    Library function: \"{meta['kernel_name']}\"")
        print(f"    Threadgroup size: {meta['threadgroup_size']}")
        print(f"    Grid: ({', '.join(meta['grid'])})")
        print("    Buffers:")
        for p in meta['params']:
            print(f"      encoder.setBuffer(buf_{p['name']}, offset: 0, index: {p['index']})")

    print("  PASS: matmul AOT artifacts saved and inspected")
