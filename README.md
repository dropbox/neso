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

Run the compile-only tests against the pinned upstream checkout with:

```sh
make test
```

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
