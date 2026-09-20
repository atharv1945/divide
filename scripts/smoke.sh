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
echo "== L_pres ablation (smoke) =="
rm -f checkpoints/smoke_ablate_lpres_on.pt checkpoints/smoke_ablate_lpres_off.pt \
      results/smoke_ablate_lpres_on_losses.csv results/smoke_ablate_lpres_on_eval.csv \
      results/smoke_ablate_lpres_off_losses.csv results/smoke_ablate_lpres_off_eval.csv \
      results/smoke_ablate_result.json
python -m src.experiments.ablate_lpres --config configs/train_pcim_cpu.yaml \
    --smoke --run-prefix smoke_ablate
echo
echo "== DBDE validation figures (smoke) =="
python -m src.experiments.dbde_validation --smoke --categories carpet bottle \
    --n-per-category 4
echo
echo "== gradio demo (construction only, no server) =="
python -c "from src.demo.app import build_demo; build_demo(); print('  demo builds OK')"
echo
echo "== deep restorer registry (fail-loud if weights absent, runs if present) =="
python -c "
from src.models.restorers import get_restorer, DEEP_SPECS
import numpy as np
img = np.clip(np.random.default_rng(0).random((32, 32, 3)).astype(np.float32), 0, 1)
for name in DEEP_SPECS:
    r = get_restorer(name)
    try:
        out = r(img)
        assert out.shape == img.shape and np.isfinite(out).all(), name
        print(f'  {name}: weights present, ran successfully')
    except RuntimeError as e:
        assert DEEP_SPECS[name]['weights_url'] in str(e), name
        print(f'  {name}: fails loudly with URL, as expected')
"
echo
echo "ALL GREEN"
