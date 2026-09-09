#!/usr/bin/env python3
"""End-to-end test: AOT compile a kernel, save .metallib, load it, run on GPU."""
import sys
import os
import struct
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neso.aot_compile import compile_kernel
from test_aot import add_kernel

# Step 1: AOT compile
print('Step 1: AOT compile add_kernel...')
result = compile_kernel(add_kernel, '*fp32, *fp32, *fp32, i32, 1024',
                        grid=['cdiv(n_elements, 1024)', '1', '1'])

# Step 2: Save to disk
tmpdir = tempfile.mkdtemp()
prefix = os.path.join(tmpdir, 'add')
print('\nStep 2: Save artifacts...')
paths = result.save(prefix)

# Step 3: Load metallib from file URL (same as Rust/Swift MTLDevice.makeLibrary(URL:))
print('\nStep 3: Load .metallib from disk...')
import Metal
from Foundation import NSURL

device = Metal.MTLCreateSystemDefaultDevice()
print(f'  Device: {device.name()}')

url = NSURL.fileURLWithPath_(paths['metallib'])
library, error = device.newLibraryWithURL_error_(url, None)
if error is not None:
    print(f'  .metallib load failed, falling back to MSL source...')
    with open(paths['metal']) as f:
        msl_source = f.read()
    options = Metal.MTLCompileOptions.alloc().init()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    library, error = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if error is not None:
        print(f'  FATAL: MSL compilation failed: {error.localizedDescription()}')
        sys.exit(1)

print(f'  Library loaded: functions={list(library.functionNames())}')

# Step 4: Create compute pipeline using metadata
print('\nStep 4: Create pipeline...')
with open(paths['json']) as f:
    meta = json.load(f)

function = library.newFunctionWithName_(meta['kernel_name'])
assert function is not None, f'Function {meta["kernel_name"]} not found'

pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
assert error is None, f'Pipeline creation failed: {error}'
print(f'  Pipeline: max_threads={pipeline.maxTotalThreadsPerThreadgroup()}')

# Step 5: Run on GPU
print('\nStep 5: Run kernel...')
N = 256
x_data = [float(i) for i in range(N)]
y_data = [float(i * 2) for i in range(N)]
n_bytes = N * 4

x_buf = device.newBufferWithBytes_length_options_(
    struct.pack(f'{N}f', *x_data), n_bytes, Metal.MTLResourceStorageModeShared)
y_buf = device.newBufferWithBytes_length_options_(
    struct.pack(f'{N}f', *y_data), n_bytes, Metal.MTLResourceStorageModeShared)
z_buf = device.newBufferWithBytes_length_options_(
    b'\x00' * n_bytes, n_bytes, Metal.MTLResourceStorageModeShared)
n_buf = device.newBufferWithBytes_length_options_(
    struct.pack('i', N), 4, Metal.MTLResourceStorageModeShared)

queue = device.newCommandQueue()
cmd = queue.commandBuffer()
enc = cmd.computeCommandEncoder()
enc.setComputePipelineState_(pipeline)

# Bind buffers in metadata order
enc.setBuffer_offset_atIndex_(x_buf, 0, 0)  # x_ptr
enc.setBuffer_offset_atIndex_(y_buf, 0, 1)  # y_ptr
enc.setBuffer_offset_atIndex_(z_buf, 0, 2)  # z_ptr
enc.setBuffer_offset_atIndex_(n_buf, 0, 3)  # n_elements

tg_size = Metal.MTLSizeMake(
    min(meta['threadgroup_size'], pipeline.maxTotalThreadsPerThreadgroup()), 1, 1)
grid_size = Metal.MTLSizeMake(1, 1, 1)
enc.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if cmd.error() is not None:
    print(f'  GPU error: {cmd.error().localizedDescription()}')
    sys.exit(1)

# Verify results
z_result = struct.unpack(f'{N}f', bytes(z_buf.contents().as_buffer(n_bytes)))
expected = [x + y for x, y in zip(x_data, y_data)]
max_err = max(abs(a - b) for a, b in zip(z_result, expected))

print(f'  z[0]={z_result[0]:.1f} (expect {expected[0]:.1f})')
print(f'  z[1]={z_result[1]:.1f} (expect {expected[1]:.1f})')
print(f'  z[255]={z_result[255]:.1f} (expect {expected[255]:.1f})')
print(f'  Max error: {max_err:.2e}')

if max_err < 1e-6:
    print('\nSUCCESS: AOT-compiled kernel produces correct results!')
else:
    print(f'\nFAIL: max error {max_err} exceeds threshold')
    sys.exit(1)

# Cleanup
shutil.rmtree(tmpdir)
