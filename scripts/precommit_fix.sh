#!/usr/bin/env bash
set -e

# Safely remove stale caches if present (ignore errors)
git rm -r --cached tests/__pycache__ || true
git rm -r --cached src/**/__pycache__ || true
git rm --cached *.pyc || true || true

# Run pre-commit hooks (auto-fix where supported)
pre-commit run --all-files

# Restage all files (including fixes)
git add -A
