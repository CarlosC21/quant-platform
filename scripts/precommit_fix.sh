#!/usr/bin/env bash
set -e

echo "Running pre-commit hooks with auto-fix..."
pre-commit run --all-files || true

echo "Fixing with ruff..."
ruff . --fix || true

echo "Applying isort..."
isort . || true

echo "Staging all changes..."
git add -A

echo "Done. Now you can commit and push."
