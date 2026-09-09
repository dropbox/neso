UV ?= uv
PYTHON ?= python3
VENV ?= env
VENV_PYTHON := $(VENV)/bin/python
UV_DEFAULT_INDEX ?= https://pypi.org/simple

export UV_PROJECT_ENVIRONMENT := $(VENV)
export UV_DEFAULT_INDEX
export VIRTUAL_ENV := $(abspath $(VENV))

PYTEST_TESTS := tests/test_aot.py tests/test_codegen.py
SCRIPT_TESTS := \
	tests/test_flash_attention.py \
	tests/test_quantized_matmul.py \
	tests/test_rel_pos_fa2.py \
	tests/test_triton_jit.py \
	tests/test_tile_ops.py \
	$(sort $(wildcard tests/test_advanced*.py))
BENCHMARKS ?= \
	bench/bench_vs_mps.py \
	bench/bench_matmul.py \
	bench/bench_flash_attention.py

INTEL_HOST ?= mac
INTEL_REPO ?= triton
INTEL_NESO_DIR ?= third_party/metal
INTEL_PYTHON ?= env/bin/python
INTEL_CORRECTNESS_TESTS ?= $(notdir $(SCRIPT_TESTS))
INTEL_BENCHMARKS ?= bench_scalar_matmul.py
WINDOWS_HOST ?= windows
WINDOWS_DIR ?= neso
WINDOWS_TARGET ?= x86_64-pc-windows-msvc
WINDOWS_KERNEL_OUT ?= target/windows-kernels
WINDOWS_BINARY := target/$(WINDOWS_TARGET)/release/neso-windows-kernels.exe
DXC_PATH ?= ../directxshadercompiler/build-release/bin/dxc
TRITON_CACHE_DIR ?= target/triton-cache

.PHONY: install sync sync-dev test-compile test test-local correctness \
	test-intel test-metal-targets windows-build test-windows test-win \
	test-targets bench benchmark bench-intel benchmark-intel \
	bench-windows benchmark-windows bench-win bench-targets \
	benchmark-targets lock

install:
	$(UV) sync --locked --python $(PYTHON) --all-extras

sync:
	$(UV) sync --locked --python $(PYTHON)

sync-dev:
	$(UV) sync --locked --python $(PYTHON) --extra runtime --extra test

test-compile: sync-dev
	$(UV) run --locked --python $(PYTHON) --extra test python -m pytest -s --tb=short \
		$(PYTEST_TESTS)

test: sync-dev
	@status=0; failed=""; \
	echo "==> pytest kernel compilation tests"; \
	$(UV) run --locked --python $(PYTHON) --extra test python -m pytest -s --tb=short \
		$(PYTEST_TESTS) || { status=1; failed="$$failed pytest"; }; \
	for test in $(SCRIPT_TESTS); do \
		echo "==> $$test"; \
		$(VENV_PYTHON) "$$test" test || { status=1; failed="$$failed $$test"; }; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED suites:$$failed"; fi; \
	exit $$status

test-local correctness: test

test-intel:
	ssh $(INTEL_HOST) 'cd $(INTEL_REPO) && set -eu; \
		for test in $(INTEL_CORRECTNESS_TESTS); do \
			echo "==> $(INTEL_NESO_DIR)/$$test"; \
			$(INTEL_PYTHON) "$(INTEL_NESO_DIR)/$$test"; \
		done'

test-metal-targets: test-local test-intel

windows-build: install
	DXC_PATH="$(DXC_PATH)" TRITON_CACHE_DIR="$(TRITON_CACHE_DIR)" \
		$(VENV_PYTHON) windows/build_kernels.py \
		--out "$(WINDOWS_KERNEL_OUT)"
	NESO_WINDOWS_KERNEL_DIR="$(abspath $(WINDOWS_KERNEL_OUT))" \
		cargo xwin build --locked --release --target $(WINDOWS_TARGET) \
		--bin neso-windows-kernels

test-windows test-win: windows-build
	ssh $(WINDOWS_HOST) 'mkdir -p $(WINDOWS_DIR)'
	ecp $(WINDOWS_BINARY) $(WINDOWS_HOST):$(WINDOWS_DIR)/
	ssh $(WINDOWS_HOST) 'cd $(WINDOWS_DIR) && ./neso-windows-kernels.exe test'

test-targets: test-local test-intel test-windows

bench benchmark: sync-dev
	@set -eu; \
	for benchmark in $(BENCHMARKS); do \
		echo "==> $$benchmark"; \
		$(VENV_PYTHON) "$$benchmark"; \
	done

bench-intel benchmark-intel:
	ssh $(INTEL_HOST) 'cd $(INTEL_REPO) && set -eu; \
		for benchmark in $(INTEL_BENCHMARKS); do \
			echo "==> $(INTEL_NESO_DIR)/$$benchmark"; \
			$(INTEL_PYTHON) "$(INTEL_NESO_DIR)/$$benchmark"; \
		done'

bench-windows benchmark-windows bench-win: windows-build
	ssh $(WINDOWS_HOST) 'mkdir -p $(WINDOWS_DIR)'
	ecp $(WINDOWS_BINARY) $(WINDOWS_HOST):$(WINDOWS_DIR)/
	ssh $(WINDOWS_HOST) 'cd $(WINDOWS_DIR) && ./neso-windows-kernels.exe bench'

bench-targets benchmark-targets: bench bench-intel bench-windows

lock:
	$(UV) lock
