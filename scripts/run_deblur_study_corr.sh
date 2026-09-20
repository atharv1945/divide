#!/usr/bin/env bash
# Recomputes defect_residual_correlation (added after the original deblur
# study ran, see src/metrics/core.py) for restormer_deblur and bilateral -
# the two rows from results/drr_study_deblur* whose >=1.0 relative DRR needs
# a shape check, not just a magnitude one. Classical wiener/classical_pipeline
# aren't re-run here - the bridging ablation (scripts/run_wiener_bridge_
# ablation.sh) already covers config A = wiener at this exact scope with the
# new column. Same scope as the original deblur study for direct
# comparability: defocus + motion, severities 2-4, 3 categories, 10 images,
# all anomaly kinds.
set -e
python -m src.experiments.drr_study \
    --families defocus motion --severities 2 3 4 --n-images 10 \
    --restorers identity bilateral restormer_deblur \
    --tag drr_study_deblur_corr --log-file results/drr_study_deblur_corr_progress.log
