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
| `make test` | Run every kernel correctness test under `tests/`. |
| `make test-compile` | Run platform-independent MSL/HLSL compilation tests. |
| `make test-local` | Alias for `make test`. |
| `make test-intel` | Run GPU correctness tests on the Intel Mac. |
| `make test-windows` | Run kernel correctness tests on Windows. |
| `make test-metal-targets` | Run the generic suite on both Metal targets. |
| `make test-targets` | Run kernel correctness tests on every target machine. |

Benchmark targets:

| Target | Purpose |
| --- | --- |
| `make bench` | Run the benchmark suite on the local Apple machine. |
| `make bench-intel` | Run scalar benchmarks on the Intel Mac. |
| `make bench-windows` | Run kernel benchmarks on Windows. |
| `make bench-targets` | Run benchmarks on Apple Silicon, Intel, and Windows. |

`make test` runs every suite even if one fails, then returns a nonzero status
if any suite failed. Application-level and speech diagnostics live under
`debug/` and are intentionally excluded.

The Windows targets compile the small kernels in `windows/kernels.py` through
Neso, compile the generated HLSL to DXIL, cross-build the local Rust harness,
and copy only that executable to the Windows machine. The harness uses
`candle-d3d12-kernels` directly; it has no model, asset, or application-level
dependencies. Set `DXC_PATH`, `WINDOWS_HOST`, or `WINDOWS_DIR` to override the
local defaults.

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
