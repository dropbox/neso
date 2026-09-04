#!/usr/bin/env python3
"""Benchmark scalar matmul on Intel/non-simdgroup GPUs."""
import sys, os, struct, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['NESO_SIMDGROUP'] = '0'
import Metal
from neso.backend.codegen import ttir_to_msl

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()
print(f'Device: {device.name()}')

def gen_matmul_ttir(BM, BN, BK):
    return f"""
    module {{
      tt.func public @matmul_kernel(
        %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>,
        %K_p: i32, %sa: i32, %sb: i32, %sc: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBK = arith.constant {BK} : i32
        %cBM = arith.constant {BM} : i32
        %cBN = arith.constant {BN} : i32
        %acc0 = arith.constant dense<0.0> : tensor<{BM}x{BN}xf32>
        %pm = tt.get_program_id x : i32
        %pn = tt.get_program_id y : i32
        %om = arith.muli %pm, %cBM : i32
        %rm = tt.make_range {{start=0:i32, end={BM}:i32}} : tensor<{BM}xi32>
        %oms = tt.splat %om : i32 -> tensor<{BM}xi32>
        %offm = arith.addi %oms, %rm : tensor<{BM}xi32>
        %on = arith.muli %pn, %cBN : i32
        %rn = tt.make_range {{start=0:i32, end={BN}:i32}} : tensor<{BN}xi32>
        %ons = tt.splat %on : i32 -> tensor<{BN}xi32>
        %offn = arith.addi %ons, %rn : tensor<{BN}xi32>
        %rk = tt.make_range {{start=0:i32, end={BK}:i32}} : tensor<{BK}xi32>
        %r = tt.expand_dims %offm {{axis=1:i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %sas = tt.splat %sa : i32 -> tensor<{BM}x1xi32>
        %ro = arith.muli %r, %sas : tensor<{BM}x1xi32>
        %ck = tt.expand_dims %rk {{axis=0:i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %rob = tt.broadcast %ro : tensor<{BM}x1xi32> -> tensor<{BM}x{BK}xi32>
        %ckb = tt.broadcast %ck : tensor<1x{BK}xi32> -> tensor<{BM}x{BK}xi32>
        %ai = arith.addi %rob, %ckb : tensor<{BM}x{BK}xi32>
        %ab = tt.splat %a : !tt.ptr<f32> -> tensor<{BM}x{BK}x!tt.ptr<f32>>
        %ap = tt.addptr %ab, %ai : tensor<{BM}x{BK}x!tt.ptr<f32>>, tensor<{BM}x{BK}xi32>
        %kr = tt.expand_dims %rk {{axis=1:i32}} : tensor<{BK}xi32> -> tensor<{BK}x1xi32>
        %sbs = tt.splat %sb : i32 -> tensor<{BK}x1xi32>
        %kro = arith.muli %kr, %sbs : tensor<{BK}x1xi32>
        %cn = tt.expand_dims %offn {{axis=0:i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %krob = tt.broadcast %kro : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
        %cnb = tt.broadcast %cn : tensor<1x{BN}xi32> -> tensor<{BK}x{BN}xi32>
        %bi = arith.addi %krob, %cnb : tensor<{BK}x{BN}xi32>
        %bb = tt.splat %b : !tt.ptr<f32> -> tensor<{BK}x{BN}x!tt.ptr<f32>>
        %bp = tt.addptr %bb, %bi : tensor<{BK}x{BN}x!tt.ptr<f32>>, tensor<{BK}x{BN}xi32>
        %adv = arith.constant dense<{BK}> : tensor<{BM}x{BK}xi32>
        %badv_s = arith.muli %sb, %cBK : i32
        %badv = tt.splat %badv_s : i32 -> tensor<{BK}x{BN}xi32>
        %res:3 = scf.for %iv = %c0 to %K_p step %cBK
            iter_args(%ap2=%ap, %bp2=%bp, %ac=%acc0)
            -> (tensor<{BM}x{BK}x!tt.ptr<f32>>, tensor<{BK}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xf32>) : i32 {{
          %la = tt.load %ap2 : tensor<{BM}x{BK}x!tt.ptr<f32>>
          %lb = tt.load %bp2 : tensor<{BK}x{BN}x!tt.ptr<f32>>
          %d = tt.dot %la, %lb, %ac : tensor<{BM}x{BK}xf32> * tensor<{BK}x{BN}xf32> -> tensor<{BM}x{BN}xf32>
          %an = tt.addptr %ap2, %adv : tensor<{BM}x{BK}x!tt.ptr<f32>>, tensor<{BM}x{BK}xi32>
          %bn = tt.addptr %bp2, %badv : tensor<{BK}x{BN}x!tt.ptr<f32>>, tensor<{BK}x{BN}xi32>
          scf.yield %an, %bn, %d : tensor<{BM}x{BK}x!tt.ptr<f32>>, tensor<{BK}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xf32>
        }}
        %cr = tt.expand_dims %offm {{axis=1:i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %scs = tt.splat %sc : i32 -> tensor<{BM}x1xi32>
        %cro = arith.muli %cr, %scs : tensor<{BM}x1xi32>
        %cc = tt.expand_dims %offn {{axis=0:i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %crob = tt.broadcast %cro : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %ccb = tt.broadcast %cc : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %ci = arith.addi %crob, %ccb : tensor<{BM}x{BN}xi32>
        %cb = tt.splat %c : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %cp = tt.addptr %cb, %ci : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        tt.store %cp, %res#2 : tensor<{BM}x{BN}x!tt.ptr<f32>>
        tt.return
      }}
    }}
    """

def bench_matmul(M, N, K, BM, BN, BK, iters=20):
    ttir = gen_matmul_ttir(BM, BN, BK)
    msl, name = ttir_to_msl(ttir, use_simdgroup=False)
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    lib, err = device.newLibraryWithSource_options_error_(msl, options, None)
    if err:
        return 0, 0, f'COMPILE: {err.localizedDescription()[:60]}'
    fn = lib.newFunctionWithName_(name)
    pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    threads = min(1024, pipe.maxTotalThreadsPerThreadgroup())

    random.seed(42)
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.gauss(0, 0.5) for _ in range(K * N)]
    a_buf = device.newBufferWithBytes_length_options_(
        struct.pack(f'{M*K}f', *a_data), M*K*4, Metal.MTLResourceStorageModeShared)
    b_buf = device.newBufferWithBytes_length_options_(
        struct.pack(f'{K*N}f', *b_data), K*N*4, Metal.MTLResourceStorageModeShared)
    c_buf = device.newBufferWithLength_options_(M*N*4, Metal.MTLResourceStorageModeShared)
    bufs = [a_buf, b_buf, c_buf] + [
        device.newBufferWithBytes_length_options_(struct.pack('i', v), 4,
            Metal.MTLResourceStorageModeShared) for v in [K, K, N, N]]
    grid = Metal.MTLSizeMake(M // BM, N // BN, 1)
    tg = Metal.MTLSizeMake(threads, 1, 1)

    # Warmup
    for _ in range(3):
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pipe)
        for i, buf in enumerate(bufs):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

    # Timed
    gpu_times = []
    for _ in range(iters):
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pipe)
        for i, buf in enumerate(bufs):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        gpu_times.append((cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0)

    gpu_times.sort()
    mn = gpu_times[len(gpu_times) // 2]
    flops = 2.0 * M * N * K
    gflops = flops / (mn / 1000.0) / 1e9 if mn > 0 else 0
    return mn, gflops, f'thr={threads}'

print()
header = f"{'Size':>14s}  {'Tile':>10s}  {'Time(ms)':>8s}  {'GFLOP/s':>9s}  Info"
print(header)
print('-' * len(header))
for M, N, K, BM, BN, BK in [
    (128, 128, 128, 32, 32, 32),
    (256, 256, 256, 32, 32, 32),
    (512, 512, 512, 32, 32, 32),
    (1024, 1024, 1024, 32, 32, 32),
    (128, 128, 128, 16, 16, 32),
    (256, 256, 256, 16, 16, 32),
    (512, 512, 512, 16, 16, 32),
    (128, 128, 128, 16, 16, 16),
    (256, 256, 256, 16, 16, 16),
    (512, 512, 512, 16, 16, 16),
    (128, 128, 128, 8, 8, 32),
    (256, 256, 256, 8, 8, 32),
    (512, 512, 512, 8, 8, 32),
]:
    t, gf, info = bench_matmul(M, N, K, BM, BN, BK)
    tile_str = f"{BM}x{BN}x{BK}"
    size_str = f"{M}x{N}x{K}"
    print(f"  {size_str:>12s}  {tile_str:>10s}  {t:>8.3f}  {gf:>9.1f}  {info}")
