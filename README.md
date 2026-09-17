# Introduction

The majority of the world's GPU are not in data centers, but sitting mostly
unutilized inside personal devices such as laptops or smartphones.
Unfortunately, it is quite hard to program these GPUs, as they use
platform-specific shader languages such as MSL or HLSL.  In the datacenter
space, Triton has become popular because it allows writing advanced kernels
like Flash Attention II in a high-devel DSL, and having a compiler specialize
for the specific GPU family and capabilities.

Neso (named after one of Neptune's moons, like Triton) aims to bridge this gap
by providing a Triton backend that lowers Triton TTIR/TTGIR to MSL or HLSL, to
make writing or porting performant GPU kernels for these platforms easier.  In
reality, these kernels might still get written by agents, but using a shared
high-level language should make code review easier and avoid unnecessary code
duplication.

For an example of a project using Neso for fast inference on MacOS and Windows,
please see https://github.com/dropbox/nspeech . A future release of Witchcraft
https://github.com/dropbox/witchcraft will also be based on Neso, which allows
us to remove the dependency on OpenVINO that we currently use to get Windows
platform support.

Apart from the compiler backend, this repo also includes basic boilerplate for
running GPU kernels on macOS and Windows. The boilerplate code depends on Rust
and Huggingface's Candle framework, in a version that we have tweaked slightly
to add Windows D3D12 support, see https://github.com/jacobgorm/candle . However,
you can use Neso in your own projects without depending on Rust or Candle.

# Neso AOT backend

This package registers the `neso` backend with Triton and compiles Triton
kernels ahead of time to Metal Shading Language or HLSL. Kernel launch support
is optional and intended only for development and benchmarking.

## Install with uv

Create and populate the project environment:

```sh
make sync
```

The Makefile sets `UV_PROJECT_ENVIRONMENT=env`, so the virtual environment is
always named `env` rather than `.venv`.

On Linux, uv installs the Triton 3.8.0 wheel from PyPI. On macOS, where Triton
does not publish a wheel, uv clones and builds the upstream revision pinned in
`pyproject.toml`. The resulting build is cached by uv, so this requires no
custom macOS wheel or CI job.

Install the optional development runtime and test dependencies when needed:

```sh
make sync-dev
```

## Make targets

Environment targets:

| Target | Purpose |
| --- | --- |
| `make install` | Install the locked project and all extras into `env`. |
| `make sync` | Install the locked base project into `env`. |
| `make sync-dev` | Install the runtime and test extras into `env`. |

Correctness targets:

| Target | Purpose |
| --- | --- |
| `make test` | Run every Python kernel test plus the local Rust/Candle checks. |
| `make test-compile` | Run platform-independent MSL/HLSL compilation tests. |
| `make test-macos` | Run the Rust/Candle kernel checks on the local Mac. |
| `make test-local` | Run the Python suite and local Rust/Candle checks. |
| `make test-intel` | Cross-build and run the Rust/Candle checks on the Intel Mac. |
| `make test-windows` | Run kernel correctness tests on Windows. |
| `make test-metal-targets` | Run the generic suite on both Metal targets. |
| `make test-targets` | Run kernel correctness tests on every target machine. |

Benchmark targets:

| Target | Purpose |
| --- | --- |
| `make bench` | Run the benchmark suite on the local Apple machine. |
| `make bench-macos` | Run the Rust/Candle kernel benchmarks on the local Mac. |
| `make bench-intel` | Cross-build and run the Rust/Candle benchmarks on the Intel Mac. |
| `make bench-windows` | Run kernel benchmarks on Windows. |
| `make bench-targets` | Run benchmarks on Apple Silicon, Intel, and Windows. |

The Python phase of `make test` runs every suite under `tests/` even if one
fails, then returns a nonzero status if any suite failed. Application-level and
speech diagnostics live under `debug/` and are intentionally excluded.

The macOS and Windows harnesses exercise the same vector-add, scale, and Flash
Attention 2 kernels with identical correctness inputs and benchmark sizes. The
FA2 benchmark uses FP16 Q/K/V on Apple Silicon and Windows, and FP32 on Intel
Metal where it avoids expensive scalar half conversions. Accumulation and
output are FP32 on every platform. The head dimension is 64, with sequence
lengths 128, 256, 512, 1024, and 2048; the harness reports median latency and
effective GFLOP/s. Neso
compiles the kernels to Metal libraries or DXIL, then Rust dispatches them
through Candle's low-level Metal or D3D12 APIs. The Intel and Windows targets
cross-build locally and copy only the executable to the target machine. They
have no model, asset, speech, or application-level dependencies.

Set `INTEL_HOST`, `INTEL_DIR`, `WINDOWS_HOST`, `WINDOWS_DIR`, or `DXC_PATH` to
override the target and toolchain defaults.

Compile a kernel from any platform with the installed command:

```sh
UV_PROJECT_ENVIRONMENT=env uv run \
  neso kernels.py \
  --kernel add_kernel \
  --signature "*fp32, *fp32, *fp32, i32, 1024" \
  --grid "cdiv(n_elements, 1024), 1, 1" \
  --output add
```

The compiler always emits source and metadata. On macOS, pass
`--require-metallib` to require Xcode's offline Metal tools and emit a compiled
`.metallib` as well.

# License

Unless otherwise noted:
```
Copyright (c) 2026 Dropbox Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
