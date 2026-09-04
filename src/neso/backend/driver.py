"""
Neso backend driver for Triton.

Handles device management, kernel loading, and compute dispatch
using Apple's Metal framework via PyObjC.
"""
from __future__ import annotations

import functools
import os

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _metal_is_available() -> bool:
    """Check if Metal is available on this system."""
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except ImportError:
        return False
    except Exception:  # PyObjC may surface Objective-C initialization failures.  # noqa: BLE001
        return False


@functools.lru_cache
def _get_metal_device():
    """Get the default Metal device."""
    import Metal
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device available")
    return device


def _get_device_name() -> str:
    """Get the Metal device name."""
    device = _get_metal_device()
    return device.name()


def _get_apple_gpu_arch() -> str:
    """Determine the Apple GPU architecture string."""
    device = _get_metal_device()
    name = device.name().lower()

    # Map device names to architecture identifiers
    if 'm4' in name:
        return 'applegpu_m4'
    elif 'm3' in name:
        return 'applegpu_m3'
    elif 'm2' in name:
        return 'applegpu_m2'
    elif 'm1' in name:
        return 'applegpu_m1'
    elif 'a17' in name:
        return 'applegpu_a17'
    elif 'a16' in name:
        return 'applegpu_a16'
    elif 'a15' in name:
        return 'applegpu_a15'
    elif 'a14' in name:
        return 'applegpu_a14'
    else:
        return 'applegpu_g15'  # generic Apple GPU family


def _get_gpu_family() -> int:
    """Get the Metal GPU family as an integer for the GPUTarget arch field."""
    device = _get_metal_device()
    name = device.name().lower()
    # Return a numeric representation for GPUTarget.arch
    # Use Apple Silicon generation number
    if 'm4' in name:
        return 4
    elif 'm3' in name:
        return 3
    elif 'm2' in name:
        return 2
    elif 'm1' in name:
        return 1
    return 1  # default


class MetalUtils:
    """Utility class for Metal operations (singleton)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        import Metal
        self._Metal = Metal
        self._device = _get_metal_device()
        self._command_queue = self._device.newCommandQueue()
        self._loaded_libraries: dict[bytes, tuple] = {}  # hash -> (library, pipeline)

    def load_binary(self, name: str, binary: bytes, shared_mem: int, device_id: int):
        """Load a compiled Metal kernel.

        Args:
            name: kernel function name
            binary: MSL source (UTF-8) or compiled .metallib bytes
            shared_mem: shared memory size (unused for Metal, threadgroup mem is declared in shader)
            device_id: device index (unused, single device)

        Returns:
            (library, pipeline_state, 0, 0, max_threads) - matching CUDA driver interface
        """
        Metal = self._Metal

        # Check if this is MSL source (UTF-8 text) or compiled metallib
        try:
            msl_source = binary.decode('utf-8')
            is_source = True
        except UnicodeDecodeError:
            is_source = False

        if is_source:
            # Runtime compilation from MSL source
            options = Metal.MTLCompileOptions.alloc().init()
            options.setFastMathEnabled_(True)
            options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)

            library, error = self._device.newLibraryWithSource_options_error_(
                msl_source, options, None
            )
            if error is not None:
                error_msg = str(error.localizedDescription())
                raise RuntimeError(
                    f"Metal MSL compilation failed:\n{error_msg}\n\nMSL source:\n{msl_source}"
                )
        else:
            # Load pre-compiled .metallib via temp file
            # (newLibraryWithData_error_ requires dispatch_data_t, not NSData,
            #  and PyObjC doesn't bridge this correctly — use URL path instead)
            import tempfile

            from Foundation import NSURL
            with tempfile.NamedTemporaryFile(suffix='.metallib', delete=False) as f:
                f.write(binary)
                lib_path = f.name
            try:
                url = NSURL.fileURLWithPath_(lib_path)
                library, error = self._device.newLibraryWithURL_error_(url, None)
                if error is not None:
                    raise RuntimeError(f"Failed to load Metal library: {error.localizedDescription()}")
            finally:
                import os as _os
                _os.unlink(lib_path)

        # Get the kernel function
        function = library.newFunctionWithName_(name)
        if function is None:
            available = list(library.functionNames())
            raise RuntimeError(
                f"Metal kernel function '{name}' not found in library. "
                f"Available functions: {available}"
            )

        # Create compute pipeline state
        pipeline_state, error = self._device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if error is not None:
            raise RuntimeError(f"Failed to create pipeline state: {error.localizedDescription()}")

        max_threads = pipeline_state.maxTotalThreadsPerThreadgroup()

        # Return format matches what CompiledKernel expects:
        # module, function, n_regs, n_spills, n_max_threads
        return library, pipeline_state, 0, 0, max_threads

    def unload_module(self, module):
        """Unload a Metal library (no-op, handled by ARC)."""

    def get_device_properties(self, device_id: int = 0) -> dict:
        """Get Metal device properties."""
        device = self._device
        return {
            "name": device.name(),
            "max_shared_mem": 32768,  # Apple GPU threadgroup memory limit
            "max_threads_per_threadgroup": 1024,
            "max_threadgroup_memory_length": device.maxThreadgroupMemoryLength(),
            "has_unified_memory": device.hasUnifiedMemory(),
        }


class MetalLauncher:
    """Launches Metal compute kernels."""

    def __init__(self, src, metadata):
        self.metadata = metadata
        self.src = src
        # Build the mapping from driver arg index -> kernel buffer index.
        # Triton passes ALL args (including constexpr and specialized-away params)
        # but the kernel only has params for the surviving TTIR function parameters.
        self._arg_indices = None
        ttir_names = getattr(metadata, 'ttir_param_names', None)
        if ttir_names and hasattr(src, 'fn') and hasattr(src.fn, 'arg_names'):
            all_names = src.fn.arg_names
            # Map each TTIR param name to its position in the original arg list
            self._arg_indices = []
            for name in ttir_names:
                try:
                    self._arg_indices.append(all_names.index(name))
                except ValueError:
                    pass  # param name not found (shouldn't happen)

    def __call__(self, gridX, gridY, gridZ, stream, pipeline_state, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        import struct

        import Metal

        if launch_enter_hook is not None and launch_metadata is not None:
            launch_enter_hook(launch_metadata.get())

        utils = MetalUtils()
        device = utils._device
        command_queue = utils._command_queue

        # Ensure MPS operations are complete before we access tensor data
        import torch
        torch.mps.synchronize()

        do_profile = os.environ.get('NESO_PROFILE')
        if do_profile:
            import time
            _t0 = time.perf_counter()

        # Filter args to match kernel parameters.
        # The driver receives ALL args including constexpr and eliminated params,
        # but the kernel only has buffers for surviving TTIR function params.
        if self._arg_indices is not None:
            kernel_args = [args[i] for i in self._arg_indices]
        else:
            kernel_args = list(args)

        # Create command buffer
        command_buffer = command_queue.commandBuffer()

        # Create compute command encoder
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        # Track tensor args and their Metal buffers for writeback
        tensor_buf_map = []  # (arg, metal_buf, buf_idx)

        # Debug: print args when NESO_DEBUG is set
        if os.environ.get('NESO_DEBUG'):
            print(f'[MetalLauncher] grid=({gridX},{gridY},{gridZ}) kernel_metadata={kernel_metadata}')
            if self._arg_indices is not None:
                print(f'  arg_indices (driver->kernel): {self._arg_indices}')
            for i, arg in enumerate(kernel_args):
                if hasattr(arg, 'data_ptr'):
                    print(f'  kernel_arg[{i}]: tensor shape={arg.shape} stride={arg.stride()} device={arg.device}')
                else:
                    print(f'  kernel_arg[{i}]: {type(arg).__name__} = {arg}')

        # Set buffer arguments
        for buf_idx, arg in enumerate(kernel_args):
            if hasattr(arg, 'data_ptr'):
                # PyTorch tensor - copy data into a shared Metal buffer
                nbytes = arg.nelement() * arg.element_size()
                # Copy tensor data through CPU to Metal shared buffer
                cpu_tensor = arg.cpu().contiguous()
                raw_bytes = bytes(cpu_tensor.untyped_storage())
                metal_buf = device.newBufferWithBytes_length_options_(
                    raw_bytes, nbytes, Metal.MTLResourceStorageModeShared
                )
                encoder.setBuffer_offset_atIndex_(metal_buf, 0, buf_idx)
                tensor_buf_map.append((arg, metal_buf, nbytes))
            elif isinstance(arg, (int, float)):
                # Scalar constant - encode as bytes in a small buffer
                if isinstance(arg, int):
                    raw = struct.pack('i', arg)
                else:
                    raw = struct.pack('f', arg)
                metal_buf = device.newBufferWithBytes_length_options_(
                    raw, len(raw), Metal.MTLResourceStorageModeShared
                )
                encoder.setBuffer_offset_atIndex_(metal_buf, 0, buf_idx)
        if do_profile:
            _t1 = time.perf_counter()

        # Determine threadgroup size and grid size
        num_warps, _num_ctas, _shared_mem = kernel_metadata
        threads_per_group = num_warps * 32  # SIMD width = 32 on Apple GPU

        max_threads = pipeline_state.maxTotalThreadsPerThreadgroup()
        threads_per_group = min(threads_per_group, max_threads)

        threadgroup_size = Metal.MTLSizeMake(threads_per_group, 1, 1)
        grid_size = Metal.MTLSizeMake(gridX, gridY, gridZ)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        # Commit and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        if command_buffer.error() is not None:
            raise RuntimeError(
                f"Metal compute kernel failed: {command_buffer.error().localizedDescription()}"
            )

        if do_profile:
            _t2 = time.perf_counter()
            # GPU-side timing from Metal command buffer
            gpu_start = command_buffer.GPUStartTime()
            gpu_end = command_buffer.GPUEndTime()
            gpu_ms = (gpu_end - gpu_start) * 1000.0

        # Debug: dump C buffer contents before copy-back
        if os.environ.get('NESO_DEBUG') and len(tensor_buf_map) >= 3:
            import struct as _struct
            _, c_buf, c_nbytes = tensor_buf_map[2]  # C is 3rd tensor
            _raw = c_buf.contents().as_buffer(min(64, c_nbytes))
            _vals = _struct.unpack(f'{min(16, c_nbytes//4)}f', bytes(_raw)[:min(64, c_nbytes)])
            print(f'[MetalLauncher] C buffer first 16 values after kernel: {_vals}')

        # Copy results back to MPS tensors
        for tensor_arg, metal_buf, nbytes in tensor_buf_map:
            result_ptr = metal_buf.contents()
            result_bytes = bytes(result_ptr.as_buffer(nbytes))
            cpu_result = torch.frombuffer(bytearray(result_bytes), dtype=tensor_arg.dtype).reshape(tensor_arg.shape)
            tensor_arg.copy_(cpu_result.to('mps'))

        if do_profile:
            _t3 = time.perf_counter()
            copy_in_ms = (_t1 - _t0) * 1000.0
            (_t2 - _t1) * 1000.0
            copy_out_ms = (_t3 - _t2) * 1000.0
            total_ms = (_t3 - _t0) * 1000.0
            print(f'[profile] copy_in={copy_in_ms:.3f}ms  '
                  f'gpu={gpu_ms:.3f}ms  '
                  f'copy_out={copy_out_ms:.3f}ms  '
                  f'total={total_ms:.3f}ms')

        if launch_exit_hook is not None and launch_metadata is not None:
            launch_exit_hook(launch_metadata.get())


def _tensor_to_metal_buffer(device, tensor):
    """Convert a tensor to a Metal buffer.

    For MPS tensors, retrieves the underlying Metal buffer.
    For CPU tensors, creates a new shared Metal buffer.
    """
    import ctypes

    import Metal

    if hasattr(tensor, 'is_mps') and tensor.is_mps:
        # For MPS backend tensors, use the objc storage
        # PyTorch MPS tensors store data in Metal buffers
        # Access via the internal MPS allocator
        ptr = tensor.data_ptr()
        nbytes = tensor.nelement() * tensor.element_size()
        buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ctypes.c_void_p(ptr), nbytes,
            Metal.MTLResourceStorageModeShared,
            None
        )
        return buf

    # For other tensors, copy data to a shared Metal buffer
    ptr = tensor.data_ptr()
    nbytes = tensor.nelement() * tensor.element_size()
    buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
        ctypes.c_void_p(ptr), nbytes,
        Metal.MTLResourceStorageModeShared,
        None
    )
    return buf


def ty_to_cpp(ty: str) -> str:
    """Map Triton type string to C++ type string."""
    if ty[0] == '*':
        return "MTLBuffer*"
    return {
        "i1": "bool",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class NesoDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @staticmethod
    def is_active():
        import os
        override = os.environ.get('TRITON_BACKEND', '')
        if override and override != 'metal':
            return False
        return _metal_is_available()

    def get_current_target(self):
        arch = _get_gpu_family()
        warp_size = 32  # Apple GPU SIMD width
        return GPUTarget("neso", arch, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("mps")

    def get_device_interface(self):
        import torch
        return torch.mps

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_device(self):
        return 0  # Single Metal device

    def get_current_stream(self, device=0):
        return 0  # Metal uses command queues, not streams
