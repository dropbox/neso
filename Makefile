UV ?= uv
PYTHON ?= python3
UV_DEFAULT_INDEX ?= https://pypi.org/simple

# uv defaults to .venv and has no project-file setting for this path.
export UV_PROJECT_ENVIRONMENT := env
export UV_DEFAULT_INDEX

.PHONY: sync
sync:
	$(UV) sync --locked --python $(PYTHON)

.PHONY: sync-dev
sync-dev:
	$(UV) sync --locked --python $(PYTHON) --extra runtime --extra test

.PHONY: test
test:
	$(UV) run --locked --python $(PYTHON) --extra test python -m pytest -s --tb=short \
		tests/test_aot.py tests/test_codegen.py

.PHONY: lock
lock:
	$(UV) lock
