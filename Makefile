PYTHON ?= python3
UV ?= uv
APP_MODULE := src.app
TEMPORAL_WORKER_MODULE := src.temporal.worker
TEMPORAL_NAMESPACE ?= default
TEMPORAL_TASK_QUEUE ?= rag0
TEMPORAL_ADDRESS ?= 127.0.0.1:7233

.PHONY: install clean interactive lint format typecheck test

install:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) pip install --editable .[dev]; \
	else \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install --editable .[dev]; \
	fi

interactive:
	$(PYTHON) -m $(APP_MODULE) --address $(TEMPORAL_ADDRESS) --namespace $(TEMPORAL_NAMESPACE) --task-queue $(TEMPORAL_TASK_QUEUE)

clean:
	rm -rf parsed storage logs .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml

lint:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run ruff check .; \
	else \
		$(PYTHON) -m ruff check .; \
	fi

format:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run ruff format .; \
	else \
		$(PYTHON) -m ruff format .; \
	fi

typecheck:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run mypy --explicit-package-bases src tests; \
	else \
		$(PYTHON) -m mypy --explicit-package-bases src tests; \
	fi

test:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run pytest; \
	else \
		$(PYTHON) -m pytest; \
	fi
