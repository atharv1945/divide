#!/usr/bin/env bash
# The learned-deblurring follow-up to the go/no-go study (results/drr_study_gonogo*).
#
# Question: does a LEARNED deblurrer erase sparse defects the way classical
# deconvolution does? The go/no-go study showed a clean mechanism split on
# scratches once two rows were excluded - deconvolution (wiener 0.335,
# classical_pipeline 0.516) erodes, denoising (restormer, nlm, gaussian,
# bilateral: 0.88-0.98) preserves - but neither deep model tested there was
# actually trained for the degradation class where erosion shows up: both
# were denoising checkpoints (NAFNet-SIDD, Restormer real_denoising).
#
# Restorer selection, and why each row is or isn't here:
#   identity                  - baseline
#   wiener                    - classical deconvolution reference (the 0.335 result)
#   classical_pipeline        - classical deconvolution reference (the 0.516 result)
#   bilateral                 - denoising control (cheap, matches the "preserves" side)
#   restormer_deblur          - the actual test: Restormer's own Motion_Deblurring /
#                                Defocus_Deblurring checkpoints (NOT real_denoising),
#                                dispatched per image by DBDE's blur_kind estimate -
#                                src/models/deep_restorers.py's
#                                build_restormer_deblur_restorer()
#   nafnet, msrcr, clahe      - DROPPED, not silently:
#     nafnet - trained on real camera sensor noise (SIDD), not synthetic blur;
#              in the go/no-go run it produced output WORSE than its own input
#              (21.11dB vs 22.84dB raw degraded, DRemR -0.70) - an
#              out-of-distribution failure, not a meaningful data point for
#              "does deblurring erase defects", and no deblurring-task NAFNet
#              checkpoint exists to substitute.
#     msrcr, clahe - contrast/tone-mapping methods, not restorers in the
#              degradation-removal sense; their drr_rel > 1 in the go/no-go
#              run reflects amplification/ringing artefacts, not defect
#              preservation - not a meaningful comparison either direction.
#
# Scope restricted to defocus/motion (the families with a matching Restormer
# deblurring checkpoint) x severities 2-4, matching the go/no-go study's
# scope otherwise (3 categories, 10 images, all anomaly kinds).
#
# Run because we don't know which way this lands, not because either answer
# is preferred - see the module docstring in drr_study.py and the go/no-go
# commit message for the full reasoning.
set -e
python -m src.experiments.drr_study \
    --families defocus motion --severities 2 3 4 --n-images 10 \
    --restorers identity wiener classical_pipeline bilateral restormer_deblur \
    --tag drr_study_deblur --log-file results/drr_study_deblur_progress.log
