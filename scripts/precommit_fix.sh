#!/usr/bin/env bash
# clean caches
git rm --cached .coverage || true
git rm -r --cached tests/__pycache__ || true

# run pre-commit hooks
pre-commit run --all-files

# stage any changes made by hooks
git add -A
