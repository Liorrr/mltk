.PHONY: install test lint fmt rust-test rust-check clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=mltk --cov-report=term-missing -q

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

rust-test:
	cd rust && cargo test

rust-check:
	cd rust && cargo fmt --check
	cd rust && cargo clippy -- -D warnings

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint test rust-check
