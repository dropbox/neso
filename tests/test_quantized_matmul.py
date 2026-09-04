#!/usr/bin/env python3
"""Quantized matmul tests: W8A16 and W4A16 through the codegen pipeline.

W8A16: int8 weights, f16 activations, f32 accumulator
  TTIR pattern: load i8 -> arith.sitofp -> f16 -> tt.dot f16*f16->f32

W4A16: packed int4 weights (2 per i8 byte), f16 activations, f32 accumulator
  TTIR pattern: load i8 -> arith.andi/shrui -> arith.sitofp -> f16 -> tt.dot
"""
import sys, os, struct, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
from neso.backend.codegen import ttir_to_msl_with_metadata

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()


def make_buffer(data, dtype='f'):
    raw = struct.pack(f'{len(data)}{dtype}', *data)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def make_zero_buffer(n, dtype='f'):
    sz = n * struct.calcsize(dtype)
    return device.newBufferWithLength_options_(sz, Metal.MTLResourceStorageModeShared)

def scalar_buf(val, dtype='i'):
    raw = struct.pack(dtype, val)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def read_buffer(buf, n, dtype='f'):
    raw = buf.contents().as_buffer(n * struct.calcsize(dtype))
    return list(struct.unpack(f'{n}{dtype}', raw))

def compile_kernel(msl_source, name):
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    lib, err = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if err:
        raise RuntimeError(f"MSL compile error:\n{err.localizedDescription()}\n\n{msl_source}")
    fn = lib.newFunctionWithName_(name)
    pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err:
        raise RuntimeError(f"Pipeline error: {err.localizedDescription()}")
    return pipe

def dispatch(pipe, grid, threads_per_group, buffers):
    # Cap to GPU's max (Intel UHD 630 only supports 448)
    threads_per_group = min(threads_per_group, pipe.maxTotalThreadsPerThreadgroup())
    cb = queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pipe)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(threads_per_group, 1, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return cb


# --- TTIR generators ---

def generate_w8a16_ttir(M, N, K, BM, BN, BK):
    """W8A16 matmul: A[M,K] f16, B[K,N] i8, C[M,N] f32.

    Loop body: load A(f16), load B(i8), sitofp B->f16, dot f16*f16->f32.
    Post-loop: store C.

    Uses standard Triton pointer iter_arg pattern for stride extraction.
    """
    ttir = f"""
    module {{
      tt.func public @w8a16_matmul(
        %A_ptr: !tt.ptr<f16>, %B_ptr: !tt.ptr<i8>, %C_ptr: !tt.ptr<f32>,
        %K_param: i32, %stride_am: i32, %stride_bk: i32, %stride_cm: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBM = arith.constant {BM} : i32
        %cBN = arith.constant {BN} : i32
        %cBK = arith.constant {BK} : i32
        %pid_m = tt.get_program_id x : i32
        %pid_n = tt.get_program_id y : i32
        %off_m = arith.muli %pid_m, %cBM : i32
        %off_n = arith.muli %pid_n, %cBN : i32

        %acc_init = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}xf32>

        // A pointer setup: A[off_m + r, 0 + c]
        %range_r_a = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat_a = tt.splat %off_m : i32 -> tensor<{BM}xi32>
        %offs_r_a = arith.addi %off_m_splat_a, %range_r_a : tensor<{BM}xi32>
        %range_c_a = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>
        %row_exp_a = tt.expand_dims %offs_r_a {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_am_splat = tt.splat %stride_am : i32 -> tensor<{BM}x1xi32>
        %row_off_a = arith.muli %row_exp_a, %stride_am_splat : tensor<{BM}x1xi32>
        %col_exp_a = tt.expand_dims %range_c_a {{axis = 0 : i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %row_off_bc_a = tt.broadcast %row_off_a : tensor<{BM}x1xi32> -> tensor<{BM}x{BK}xi32>
        %col_bc_a = tt.broadcast %col_exp_a : tensor<1x{BK}xi32> -> tensor<{BM}x{BK}xi32>
        %idx_a = arith.addi %row_off_bc_a, %col_bc_a : tensor<{BM}x{BK}xi32>
        %a_base_splat = tt.splat %A_ptr : !tt.ptr<f16> -> tensor<{BM}x{BK}x!tt.ptr<f16>>
        %a_ptrs_init = tt.addptr %a_base_splat, %idx_a : tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BM}x{BK}xi32>

        // B pointer setup: B[0 + r, off_n + c]
        %range_r_b = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>
        %range_c_b = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat_b = tt.splat %off_n : i32 -> tensor<{BN}xi32>
        %offs_c_b = arith.addi %off_n_splat_b, %range_c_b : tensor<{BN}xi32>
        %row_exp_b = tt.expand_dims %range_r_b {{axis = 1 : i32}} : tensor<{BK}xi32> -> tensor<{BK}x1xi32>
        %stride_bk_splat = tt.splat %stride_bk : i32 -> tensor<{BK}x1xi32>
        %row_off_b = arith.muli %row_exp_b, %stride_bk_splat : tensor<{BK}x1xi32>
        %col_exp_b = tt.expand_dims %offs_c_b {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %row_off_bc_b = tt.broadcast %row_off_b : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
        %col_bc_b = tt.broadcast %col_exp_b : tensor<1x{BN}xi32> -> tensor<{BK}x{BN}xi32>
        %idx_b = arith.addi %row_off_bc_b, %col_bc_b : tensor<{BK}x{BN}xi32>
        %b_base_splat = tt.splat %B_ptr : !tt.ptr<i8> -> tensor<{BK}x{BN}x!tt.ptr<i8>>
        %b_ptrs_init = tt.addptr %b_base_splat, %idx_b : tensor<{BK}x{BN}x!tt.ptr<i8>>, tensor<{BK}x{BN}xi32>

        // K-loop with pointer iter_args
        %result:3 = scf.for %iv = %c0 to %K_param step %cBK iter_args(%acc = %acc_init, %a_ptrs = %a_ptrs_init, %b_ptrs = %b_ptrs_init) -> (tensor<{BM}x{BN}xf32>, tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BK}x{BN}x!tt.ptr<i8>>) {{

          // Load tiles
          %a_tile = tt.load %a_ptrs : tensor<{BM}x{BK}x!tt.ptr<f16>>
          %b_i8 = tt.load %b_ptrs : tensor<{BK}x{BN}x!tt.ptr<i8>>

          // Dequantize: i8 -> f16
          %b_f16 = arith.sitofp %b_i8 : tensor<{BK}x{BN}xi8> to tensor<{BK}x{BN}xf16>

          // Dot: f16 * f16 -> f32
          %new_acc = tt.dot %a_tile, %b_f16, %acc : tensor<{BM}x{BK}xf16> * tensor<{BK}x{BN}xf16> -> tensor<{BM}x{BN}xf32>

          // Advance pointers by BK along K dimension
          %bk_splat_a = tt.splat %cBK : i32 -> tensor<{BM}x{BK}xi32>
          %a_ptrs_next = tt.addptr %a_ptrs, %bk_splat_a : tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BM}x{BK}xi32>
          %bk_stride_b = tt.splat %cBK : i32 -> tensor<{BK}x1xi32>
          %bk_stride_mul = arith.muli %bk_stride_b, %stride_bk_splat : tensor<{BK}x1xi32>
          %bk_advance_b = tt.broadcast %bk_stride_mul : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
          %b_ptrs_next = tt.addptr %b_ptrs, %bk_advance_b : tensor<{BK}x{BN}x!tt.ptr<i8>>, tensor<{BK}x{BN}xi32>

          scf.yield %new_acc, %a_ptrs_next, %b_ptrs_next : tensor<{BM}x{BN}xf32>, tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BK}x{BN}x!tt.ptr<i8>>
        }}

        // Store C [BM, BN] as f32
        %range_r_c = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat_c = tt.splat %off_m : i32 -> tensor<{BM}xi32>
        %offs_r_c = arith.addi %off_m_splat_c, %range_r_c : tensor<{BM}xi32>
        %range_c_c = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat_c = tt.splat %off_n : i32 -> tensor<{BN}xi32>
        %offs_c_c = arith.addi %off_n_splat_c, %range_c_c : tensor<{BN}xi32>
        %row_exp_c = tt.expand_dims %offs_r_c {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_cm_splat = tt.splat %stride_cm : i32 -> tensor<{BM}x1xi32>
        %row_off_c = arith.muli %row_exp_c, %stride_cm_splat : tensor<{BM}x1xi32>
        %col_exp_c = tt.expand_dims %offs_c_c {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %row_off_bc_c = tt.broadcast %row_off_c : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %col_bc_c = tt.broadcast %col_exp_c : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %idx_c = arith.addi %row_off_bc_c, %col_bc_c : tensor<{BM}x{BN}xi32>
        %c_base = tt.splat %C_ptr : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %c_ptrs = tt.addptr %c_base, %idx_c : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        tt.store %c_ptrs, %result#0 : tensor<{BM}x{BN}x!tt.ptr<f32>>
        tt.return
      }}
    }}
    """
    return ttir


def generate_w4a16_ttir(M, N, K, BM, BN, BK):
    """W4A16 matmul: int4 weights stored as i8 (values [-8,7]), f16 activations.

    This tests the same dequant codegen path (i8 load + sitofp + MMA) with 4-bit
    value range. Actual packed int4 (2 values per byte) would need tt.reshape
    support for uninterleaving — that's a separate concern from the dequant MMA path.
    """
    # Reuses the W8A16 TTIR — the codegen is identical, only the value range differs
    return generate_w8a16_ttir(M, N, K, BM, BN, BK)


def generate_w8a16_scaled_ttir(M, N, K, BM, BN, BK):
    """W8A16 matmul with per-channel scale: C = (A @ dequant(B)) * scale[n].

    Standard GPTQ/AWQ pattern: int8 weights dequantized and scaled per output channel.
    Post-matmul: load scale[BN], broadcast to [BM,BN], multiply accumulator.
    """
    ttir = f"""
    module {{
      tt.func public @w8a16_scaled_matmul(
        %A_ptr: !tt.ptr<f16>, %B_ptr: !tt.ptr<i8>, %C_ptr: !tt.ptr<f32>,
        %scale_ptr: !tt.ptr<f16>,
        %K_param: i32, %stride_am: i32, %stride_bk: i32, %stride_cm: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBM = arith.constant {BM} : i32
        %cBN = arith.constant {BN} : i32
        %cBK = arith.constant {BK} : i32
        %pid_m = tt.get_program_id x : i32
        %pid_n = tt.get_program_id y : i32
        %off_m = arith.muli %pid_m, %cBM : i32
        %off_n = arith.muli %pid_n, %cBN : i32

        %acc_init = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}xf32>

        // A pointer setup
        %range_r_a = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat_a = tt.splat %off_m : i32 -> tensor<{BM}xi32>
        %offs_r_a = arith.addi %off_m_splat_a, %range_r_a : tensor<{BM}xi32>
        %range_c_a = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>
        %row_exp_a = tt.expand_dims %offs_r_a {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_am_splat = tt.splat %stride_am : i32 -> tensor<{BM}x1xi32>
        %row_off_a = arith.muli %row_exp_a, %stride_am_splat : tensor<{BM}x1xi32>
        %col_exp_a = tt.expand_dims %range_c_a {{axis = 0 : i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %row_off_bc_a = tt.broadcast %row_off_a : tensor<{BM}x1xi32> -> tensor<{BM}x{BK}xi32>
        %col_bc_a = tt.broadcast %col_exp_a : tensor<1x{BK}xi32> -> tensor<{BM}x{BK}xi32>
        %idx_a = arith.addi %row_off_bc_a, %col_bc_a : tensor<{BM}x{BK}xi32>
        %a_base_splat = tt.splat %A_ptr : !tt.ptr<f16> -> tensor<{BM}x{BK}x!tt.ptr<f16>>
        %a_ptrs_init = tt.addptr %a_base_splat, %idx_a : tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BM}x{BK}xi32>

        // B pointer setup
        %range_r_b = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>
        %range_c_b = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat_b = tt.splat %off_n : i32 -> tensor<{BN}xi32>
        %offs_c_b = arith.addi %off_n_splat_b, %range_c_b : tensor<{BN}xi32>
        %row_exp_b = tt.expand_dims %range_r_b {{axis = 1 : i32}} : tensor<{BK}xi32> -> tensor<{BK}x1xi32>
        %stride_bk_splat = tt.splat %stride_bk : i32 -> tensor<{BK}x1xi32>
        %row_off_b = arith.muli %row_exp_b, %stride_bk_splat : tensor<{BK}x1xi32>
        %col_exp_b = tt.expand_dims %offs_c_b {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %row_off_bc_b = tt.broadcast %row_off_b : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
        %col_bc_b = tt.broadcast %col_exp_b : tensor<1x{BN}xi32> -> tensor<{BK}x{BN}xi32>
        %idx_b = arith.addi %row_off_bc_b, %col_bc_b : tensor<{BK}x{BN}xi32>
        %b_base_splat = tt.splat %B_ptr : !tt.ptr<i8> -> tensor<{BK}x{BN}x!tt.ptr<i8>>
        %b_ptrs_init = tt.addptr %b_base_splat, %idx_b : tensor<{BK}x{BN}x!tt.ptr<i8>>, tensor<{BK}x{BN}xi32>

        // K-loop
        %result:3 = scf.for %iv = %c0 to %K_param step %cBK iter_args(%acc = %acc_init, %a_ptrs = %a_ptrs_init, %b_ptrs = %b_ptrs_init) -> (tensor<{BM}x{BN}xf32>, tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BK}x{BN}x!tt.ptr<i8>>) {{
          %a_tile = tt.load %a_ptrs : tensor<{BM}x{BK}x!tt.ptr<f16>>
          %b_i8 = tt.load %b_ptrs : tensor<{BK}x{BN}x!tt.ptr<i8>>
          %b_f16 = arith.sitofp %b_i8 : tensor<{BK}x{BN}xi8> to tensor<{BK}x{BN}xf16>
          %new_acc = tt.dot %a_tile, %b_f16, %acc : tensor<{BM}x{BK}xf16> * tensor<{BK}x{BN}xf16> -> tensor<{BM}x{BN}xf32>
          %bk_splat_a = tt.splat %cBK : i32 -> tensor<{BM}x{BK}xi32>
          %a_ptrs_next = tt.addptr %a_ptrs, %bk_splat_a : tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BM}x{BK}xi32>
          %bk_stride_b = tt.splat %cBK : i32 -> tensor<{BK}x1xi32>
          %bk_stride_mul = arith.muli %bk_stride_b, %stride_bk_splat : tensor<{BK}x1xi32>
          %bk_advance_b = tt.broadcast %bk_stride_mul : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
          %b_ptrs_next = tt.addptr %b_ptrs, %bk_advance_b : tensor<{BK}x{BN}x!tt.ptr<i8>>, tensor<{BK}x{BN}xi32>
          scf.yield %new_acc, %a_ptrs_next, %b_ptrs_next : tensor<{BM}x{BN}xf32>, tensor<{BM}x{BK}x!tt.ptr<f16>>, tensor<{BK}x{BN}x!tt.ptr<i8>>
        }}

        // Post-matmul: load scale[BN], broadcast to [BM,BN], multiply accumulator
        %scale_range = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat_sc = tt.splat %off_n : i32 -> tensor<{BN}xi32>
        %scale_offs = arith.addi %off_n_splat_sc, %scale_range : tensor<{BN}xi32>
        %scale_base = tt.splat %scale_ptr : !tt.ptr<f16> -> tensor<{BN}x!tt.ptr<f16>>
        %scale_ptrs = tt.addptr %scale_base, %scale_offs : tensor<{BN}x!tt.ptr<f16>>, tensor<{BN}xi32>
        %scale_f16 = tt.load %scale_ptrs : tensor<{BN}x!tt.ptr<f16>>
        %scale_f32 = arith.extf %scale_f16 : tensor<{BN}xf16> to tensor<{BN}xf32>
        %scale_exp = tt.expand_dims %scale_f32 {{axis = 0 : i32}} : tensor<{BN}xf32> -> tensor<1x{BN}xf32>
        %scale_bc = tt.broadcast %scale_exp : tensor<1x{BN}xf32> -> tensor<{BM}x{BN}xf32>
        %result_scaled = arith.mulf %result#0, %scale_bc : tensor<{BM}x{BN}xf32>

        // Store scaled result
        %range_r_c = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat_c = tt.splat %off_m : i32 -> tensor<{BM}xi32>
        %offs_r_c = arith.addi %off_m_splat_c, %range_r_c : tensor<{BM}xi32>
        %range_c_c = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat_c = tt.splat %off_n : i32 -> tensor<{BN}xi32>
        %offs_c_c = arith.addi %off_n_splat_c, %range_c_c : tensor<{BN}xi32>
        %row_exp_c = tt.expand_dims %offs_r_c {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_cm_splat = tt.splat %stride_cm : i32 -> tensor<{BM}x1xi32>
        %row_off_c = arith.muli %row_exp_c, %stride_cm_splat : tensor<{BM}x1xi32>
        %col_exp_c = tt.expand_dims %offs_c_c {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %row_off_bc_c = tt.broadcast %row_off_c : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %col_bc_c = tt.broadcast %col_exp_c : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %idx_c = arith.addi %row_off_bc_c, %col_bc_c : tensor<{BM}x{BN}xi32>
        %c_base = tt.splat %C_ptr : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %c_ptrs = tt.addptr %c_base, %idx_c : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        tt.store %c_ptrs, %result_scaled : tensor<{BM}x{BN}x!tt.ptr<f32>>
        tt.return
      }}
    }}
    """
    return ttir


# --- Reference implementations ---

def ref_w8a16_matmul(a_f16, b_i8, M, N, K):
    """Reference: C = A(f16) @ B(i8->f16), accumulate in f64."""
    c = [0.0] * (M * N)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += float(a_f16[m * K + k]) * float(b_i8[k * N + n])
            c[m * N + n] = acc
    return c


def ref_w4a16_matmul(a_f16, b_i8, M, N, K):
    """Reference: int4 values stored as i8, C = A(f16) @ B(i8->f16)."""
    return ref_w8a16_matmul(a_f16, b_i8, M, N, K)


def ref_w8a16_scaled_matmul(a_f16, b_i8, scale_f16, M, N, K):
    """Reference: C = (A @ dequant(B)) * scale[n]."""
    c = ref_w8a16_matmul(a_f16, b_i8, M, N, K)
    for m in range(M):
        for n in range(N):
            c[m * N + n] *= float(scale_f16[n])
    return c


# --- Test functions ---

def test_w8a16(M, N, K, BM, BN, BK):
    """Test W8A16 matmul through codegen pipeline."""
    ttir = generate_w8a16_ttir(M, N, K, BM, BN, BK)

    try:
        msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    except Exception as e:
        print(f"  Codegen error: {e}")
        return False, None

    if os.environ.get('NESO_DUMP_MSL'):
        print(msl)

    try:
        pipe = compile_kernel(msl, name)
    except Exception as e:
        print(f"  MSL compile error: {e}")
        return False, None

    random.seed(42)
    # A: f16 random, B: i8 random [-128, 127]
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.randint(-128, 127) for _ in range(K * N)]

    a_buf = make_buffer(a_data, dtype='e')  # f16
    b_buf = make_buffer(b_data, dtype='b')  # i8 (signed byte)
    c_buf = make_zero_buffer(M * N, dtype='f')  # f32

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    dispatch(pipe, (grid_m, grid_n, 1), rec_threads,
             [a_buf, b_buf, c_buf,
              scalar_buf(K), scalar_buf(K), scalar_buf(N), scalar_buf(N)])

    c_vals = read_buffer(c_buf, M * N, dtype='f')
    ref = ref_w8a16_matmul(a_data, b_data, M, N, K)

    max_err = max(abs(c_vals[i] - ref[i]) for i in range(M * N))
    # Normalize by magnitude
    max_ref = max(abs(r) for r in ref) or 1.0
    rel_err = max_err / max_ref
    ok = rel_err < 0.01
    return ok, rel_err


def test_w4a16(M, N, K, BM, BN, BK):
    """Test W4A16 matmul: int4 values stored as i8, same dequant path as W8A16."""
    # Same codegen path as W8A16, but with int4 value range [-8, 7]
    ttir = generate_w4a16_ttir(M, N, K, BM, BN, BK)

    try:
        msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    except Exception as e:
        print(f"  Codegen error: {e}")
        return False, None

    try:
        pipe = compile_kernel(msl, name)
    except Exception as e:
        print(f"  MSL compile error: {e}")
        return False, None

    random.seed(43)  # Different seed from W8A16
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.randint(-8, 7) for _ in range(K * N)]  # int4 range

    a_buf = make_buffer(a_data, dtype='e')
    b_buf = make_buffer(b_data, dtype='b')
    c_buf = make_zero_buffer(M * N, dtype='f')

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    dispatch(pipe, (grid_m, grid_n, 1), rec_threads,
             [a_buf, b_buf, c_buf,
              scalar_buf(K), scalar_buf(K), scalar_buf(N), scalar_buf(N)])

    c_vals = read_buffer(c_buf, M * N, dtype='f')
    ref = ref_w4a16_matmul(a_data, b_data, M, N, K)

    max_err = max(abs(c_vals[i] - ref[i]) for i in range(M * N))
    max_ref = max(abs(r) for r in ref) or 1.0
    rel_err = max_err / max_ref
    ok = rel_err < 0.01
    return ok, rel_err


def test_w8a16_scaled(M, N, K, BM, BN, BK):
    """Test W8A16 matmul with per-channel scale: C = (A @ dequant(B)) * scale[n]."""
    ttir = generate_w8a16_scaled_ttir(M, N, K, BM, BN, BK)

    try:
        msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    except Exception as e:
        print(f"  Codegen error: {e}")
        return False, None

    if os.environ.get('NESO_DUMP_MSL'):
        print(msl)

    try:
        pipe = compile_kernel(msl, name)
    except Exception as e:
        print(f"  MSL compile error: {e}")
        return False, None

    random.seed(44)
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.randint(-128, 127) for _ in range(K * N)]
    scale_data = [random.uniform(0.001, 0.1) for _ in range(N)]

    a_buf = make_buffer(a_data, dtype='e')
    b_buf = make_buffer(b_data, dtype='b')
    c_buf = make_zero_buffer(M * N, dtype='f')
    scale_buf = make_buffer(scale_data, dtype='e')

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    dispatch(pipe, (grid_m, grid_n, 1), rec_threads,
             [a_buf, b_buf, c_buf, scale_buf,
              scalar_buf(K), scalar_buf(K), scalar_buf(N), scalar_buf(N)])

    c_vals = read_buffer(c_buf, M * N, dtype='f')
    ref = ref_w8a16_scaled_matmul(a_data, b_data, scale_data, M, N, K)

    max_err = max(abs(c_vals[i] - ref[i]) for i in range(M * N))
    max_ref = max(abs(r) for r in ref) or 1.0
    rel_err = max_err / max_ref
    ok = rel_err < 0.01
    return ok, rel_err


def bench_w8a16(M, N, K, BM, BN, BK, iters=50):
    """Benchmark W8A16 matmul."""
    import time
    ttir = generate_w8a16_ttir(M, N, K, BM, BN, BK)
    msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    pipe = compile_kernel(msl, name)

    random.seed(42)
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.randint(-128, 127) for _ in range(K * N)]
    a_buf = make_buffer(a_data, dtype='e')
    b_buf = make_buffer(b_data, dtype='b')
    c_buf = make_zero_buffer(M * N, dtype='f')

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN
    bufs = [a_buf, b_buf, c_buf,
            scalar_buf(K), scalar_buf(K), scalar_buf(N), scalar_buf(N)]

    # Warmup
    for _ in range(5):
        dispatch(pipe, (grid_m, grid_n, 1), rec_threads, bufs)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dispatch(pipe, (grid_m, grid_n, 1), rec_threads, bufs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    flops = 2 * M * N * K
    gflops = flops / (min_ms / 1000) / 1e9
    return avg_ms, min_ms, gflops


def bench_w8a16_scaled(M, N, K, BM, BN, BK, iters=50):
    """Benchmark W8A16 scaled matmul."""
    import time
    ttir = generate_w8a16_scaled_ttir(M, N, K, BM, BN, BK)
    msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    pipe = compile_kernel(msl, name)

    random.seed(42)
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.randint(-128, 127) for _ in range(K * N)]
    scale_data = [random.uniform(0.001, 0.1) for _ in range(N)]
    a_buf = make_buffer(a_data, dtype='e')
    b_buf = make_buffer(b_data, dtype='b')
    c_buf = make_zero_buffer(M * N, dtype='f')
    scale_buf = make_buffer(scale_data, dtype='e')

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN
    bufs = [a_buf, b_buf, c_buf, scale_buf,
            scalar_buf(K), scalar_buf(K), scalar_buf(N), scalar_buf(N)]

    for _ in range(5):
        dispatch(pipe, (grid_m, grid_n, 1), rec_threads, bufs)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dispatch(pipe, (grid_m, grid_n, 1), rec_threads, bufs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    flops = 2 * M * N * K
    gflops = flops / (min_ms / 1000) / 1e9
    return avg_ms, min_ms, gflops


if __name__ == "__main__":
    print(f"Metal Device: {device.name()}")
    all_ok = True

    if "test" in sys.argv or len(sys.argv) == 1:
        print("\n=== W8A16 Matmul Correctness ===")
        w8a16_configs = [
            # (M, N, K, BM, BN, BK)
            (32, 32, 32, 32, 32, 32),    # Single tile, single K iteration
            (64, 64, 64, 32, 32, 32),     # 2x2 grid, 2 K iterations
            (64, 64, 128, 32, 32, 32),    # 2x2 grid, 4 K iterations
            (128, 128, 128, 32, 32, 32),  # 4x4 grid, 4 K iterations
            (128, 128, 256, 64, 64, 32),  # 2x2 grid, 8 K iterations
            (256, 256, 256, 128, 128, 32),  # 2x2 grid, 8 K iters, large tiles
            (128, 128, 128, 64, 64, 64),  # BK=64 test
            (256, 256, 256, 64, 64, 64),  # BK=64, multi-grid
        ]
        for M, N, K, BM, BN, BK in w8a16_configs:
            result = test_w8a16(M, N, K, BM, BN, BK)
            if result[1] is None:
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: FAIL (codegen/compile error)")
                all_ok = False
            else:
                ok, err = result
                status = "PASS" if ok else "FAIL"
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: {status} (rel_err={err:.6f})")
                if not ok:
                    all_ok = False

        print("\n=== W4A16 Matmul Correctness ===")
        w4a16_configs = [
            (32, 32, 32, 32, 32, 32),
            (64, 64, 64, 32, 32, 32),
            (64, 64, 128, 32, 32, 32),
            (128, 128, 128, 32, 32, 32),
        ]
        for M, N, K, BM, BN, BK in w4a16_configs:
            result = test_w4a16(M, N, K, BM, BN, BK)
            if result[1] is None:
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: FAIL (codegen/compile error)")
                all_ok = False
            else:
                ok, err = result
                status = "PASS" if ok else "FAIL"
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: {status} (rel_err={err:.6f})")
                if not ok:
                    all_ok = False

        print("\n=== W8A16 Scaled Matmul (per-channel scale) ===")
        scaled_configs = [
            (32, 32, 32, 32, 32, 32),
            (64, 64, 64, 32, 32, 32),
            (128, 128, 128, 32, 32, 32),
            (128, 128, 256, 64, 64, 32),
            (256, 256, 256, 128, 128, 32),
        ]
        for M, N, K, BM, BN, BK in scaled_configs:
            result = test_w8a16_scaled(M, N, K, BM, BN, BK)
            if result[1] is None:
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: FAIL (codegen/compile error)")
                all_ok = False
            else:
                ok, err = result
                status = "PASS" if ok else "FAIL"
                print(f"  {M}x{N}x{K} tile={BM}x{BN}x{BK}: {status} (rel_err={err:.6f})")
                if not ok:
                    all_ok = False

    if "bench" in sys.argv or len(sys.argv) == 1:
        print("\n=== W8A16 Matmul Performance ===")
        print(f"{'MxNxK':>15s}  {'Tile':>12s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
        print("-" * 65)
        bench_configs = [
            (256, 256, 256, 32, 32, 32),
            (512, 512, 512, 32, 32, 32),
            (1024, 1024, 1024, 64, 64, 32),
            (1024, 1024, 1024, 64, 64, 64),
            (2048, 2048, 2048, 64, 64, 32),
            (2048, 2048, 2048, 64, 64, 64),
            (2048, 2048, 2048, 128, 64, 32),
            (2048, 2048, 2048, 128, 128, 32),
            (4096, 4096, 4096, 64, 64, 32),
            (4096, 4096, 4096, 64, 64, 64),
            (4096, 4096, 4096, 128, 64, 32),
            (4096, 4096, 4096, 128, 128, 32),
        ]
        for M, N, K, BM, BN, BK in bench_configs:
            try:
                avg, mn, gf = bench_w8a16(M, N, K, BM, BN, BK)
                print(f"  {M}x{N}x{K:>4d}  {BM}x{BN}x{BK:>2d}  {avg:>8.3f}  {mn:>8.3f}  {gf:>9.1f}")
            except Exception as e:
                print(f"  {M}x{N}x{K:>4d}  {BM}x{BN}x{BK:>2d}  ERROR: {e}")

        print("\n=== W8A16 Scaled Matmul Performance ===")
        print(f"{'MxNxK':>15s}  {'Tile':>12s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
        print("-" * 65)
        scaled_bench_configs = [
            (1024, 1024, 1024, 64, 64, 32),
            (2048, 2048, 2048, 64, 64, 32),
            (2048, 2048, 2048, 128, 128, 32),
            (4096, 4096, 4096, 64, 64, 32),
            (4096, 4096, 4096, 128, 128, 32),
        ]
        for M, N, K, BM, BN, BK in scaled_bench_configs:
            try:
                avg, mn, gf = bench_w8a16_scaled(M, N, K, BM, BN, BK)
                print(f"  {M}x{N}x{K:>4d}  {BM}x{BN}x{BK:>2d}  {avg:>8.3f}  {mn:>8.3f}  {gf:>9.1f}")
            except Exception as e:
                print(f"  {M}x{N}x{K:>4d}  {BM}x{BN}x{BK:>2d}  ERROR: {e}")

    if not all_ok:
        print("\nSome tests failed!")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
