#!/usr/bin/env bash
# Verify the entire pipeline on CPU with no dataset present.
# Run this before every push, and before any GPU session.
set -e
echo "== unit tests =="
python -m pytest tests/ -q
echo
echo "== DRR study (smoke) =="
python -m src.experiments.drr_study --smoke --n-images 4 \
    --families defocus noise --severities 3 \
    --restorers identity gaussian nlm clahe wiener --tag smoke
echo
echo "ALL GREEN"
