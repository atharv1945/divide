#!/usr/bin/env bash
# Bridges classical Wiener deconvolution to PCIM's x_cons one variable at a
# time - see src/models/wiener_ablation.py's module docstring for the full
# mechanistic rationale (testing whether classical Wiener's severe defect
# erosion, and x_cons's absence of it, comes down to regularization
# strength rather than architecture).
#
#   identity                - baseline
#   wiener                  - config A: classical Wiener, flat nsr=0.01
#   wiener_dbde_nsr         - config B: + nsr derived from DBDE's floored sigma
#   wiener_hqs_rawpixel     - config C: + unrolled HQS (growing rho), raw pixel domain
#   wiener_hqs_vst          - config D: + VST/illumination domain (= x_cons, nsr_floor=0.01)
#   wiener_hqs_vst_nofloor  - config E: x_cons with nsr_floor disabled (reverse direction)
#
# Same scope as the deblur study (results/drr_study_deblur*) for direct
# comparability: defocus + motion, severities 2-4, 3 categories, 10 images,
# all anomaly kinds.
set -e
python -m src.experiments.drr_study \
    --families defocus motion --severities 2 3 4 --n-images 10 \
    --restorers identity wiener wiener_dbde_nsr wiener_hqs_rawpixel wiener_hqs_vst wiener_hqs_vst_nofloor \
    --tag wiener_bridge_ablation --log-file results/wiener_bridge_ablation_progress.log
