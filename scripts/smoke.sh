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
echo "== eval grid (smoke) =="
python -m src.experiments.eval_grid --smoke --detectors padim \
    --restorers identity gaussian --families noise --severities 3 \
    --n-train 6 --n-test 4 --categories carpet --tag smoke_eval_grid
echo
echo "== PCIM training (smoke, incl. kill-and-resume) =="
rm -f checkpoints/smoke_train.pt results/smoke_train_losses.csv results/smoke_train_eval.csv
python -m src.experiments.train_pcim --config configs/train_pcim_cpu.yaml \
    --smoke --run-name smoke_train
echo
echo "== gradio demo (construction only, no server) =="
python -c "from src.demo.app import build_demo; build_demo(); print('  demo builds OK')"
echo
echo "== deep restorer registry (fail-loud check, no weights present) =="
python -c "
from src.models.restorers import get_restorer, DEEP_SPECS
import numpy as np
img = np.zeros((16, 16, 3), np.float32)
for name in DEEP_SPECS:
    r = get_restorer(name)
    try:
        r(img)
        raise SystemExit(f'{name}: expected RuntimeError for missing weights, got none')
    except RuntimeError as e:
        assert DEEP_SPECS[name]['weights_url'] in str(e), name
        print(f'  {name}: fails loudly with URL, as expected')
"
echo
echo "ALL GREEN"
