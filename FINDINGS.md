# FINDINGS

Digital Image Processing (BCSE403L), VIT Vellore. This document reports two
things at different levels of confidence, deliberately kept separate:

1. **The L_pres claim — established.** A 150-example held-out bootstrap
   comparison of the two final L_pres-ablation checkpoints.
2. **The detection-grid results — mixed confidence, reported honestly.**
   A 360-cell frozen-detector grid, gated by a harness-correctness check and
   a bootstrap CI on every headline number, because the first pass produced
   a result (PatchCore clean AUROC 20 points below published) that turned
   out to be a data-scale artifact, not a real detector finding — and
   several of the grid's apparent effects do not survive resampling.

Every number below is in a committed CSV/JSON under `results/`, traceable to
the script that produced it. No number here was estimated or interpolated.

---

## 1. The L_pres claim (established)

**Question:** does the preservation loss `L_pres` do anything beyond what
PCIM's `x_cons` structural path already provides?

**Method:** `src/experiments/ablate_lpres.py` trained PCIM twice — identical
seed, data order, initialization, step count — differing in exactly
`loss.use_lpres` (asserted in code, not just by convention:
`_assert_only_use_lpres_differs`). Both final checkpoints
(`checkpoints/ablate_lpres_lpres_on.pt` / `_off.pt`) were then re-evaluated,
**inference only, no retraining**, on the full 150-example held-out set
`eval_pcim_holdout.py` uses — the ablation's own 24-example eval set was too
small to trust a per-kind breakdown on. 2000 paired bootstrap resamples
(`src/experiments/compare_lpres_bootstrap.py`, resampling example indices,
the same draw applied to both checkpoints since they were scored on
identical examples) give a 95% CI on the ON−OFF delta for relative DRR,
residual correlation, and PSNR, overall and per anomaly kind.

**Result** (`results/ablate_lpres_bootstrap.json`):

| kind | n | relative DRR OFF→ON | Δ (95% CI) | residual corr OFF→ON | Δ (95% CI) | PSNR OFF→ON |
|---|---|---|---|---|---|---|
| overall | 150 | 0.913 → 1.077 | +0.164 [0.064, 0.248] | 0.542 → 0.813 | +0.271 [0.213, 0.329] | 26.33 → 24.69 dB |
| scratch | 38 | 0.687 → 1.045 | +0.358 [0.294, 0.428] | 0.532 → 0.771 | **+0.239 [0.188, 0.296]** | 25.27 → 23.71 dB |
| blob | 56 | 0.760 → 1.060 | +0.300 [0.210, 0.388] | 0.545 → 0.940 | +0.394 [0.283, 0.511] | 27.17 → 25.71 dB |
| texture | 56 | 1.261 → 1.122 | −0.139 [−0.393, 0.067] (straddles 0) | 0.545 → 0.716 | +0.171 [0.083, 0.255] | 26.20 → 24.33 dB |

**The decisive number — scratch residual-correlation delta, CI [0.188, 0.296] — excludes zero. The claim is established at this sample size**, not merely suggestive: L_pres produces a real, statistically supported increase in residual *shape* fidelity for scratches, the anomaly kind this project cares about most, not just magnitude (DRR). Overall and blob show the same pattern, also established (every listed delta above except texture's DRR has a CI excluding zero).

**Texture is not a contradiction, read correctly.** OFF→ON, texture's DRR moves from 1.261 toward 1.122 — *closer* to the faithful value of 1.0, i.e. less excess/ringing energy — while its residual correlation rises 0.545→0.716, an established increase. Per this project's own established finding (`README.md`, "The claim"), DRR above ~1.0 includes excess energy and residual correlation is the trustworthy shape measure; the correlation component of texture's story is established even though the DRR component alone isn't independently significant at n=56. Same mechanism as scratch and blob — L_pres pulling an over-amplified, distorted residual toward a smaller, more correctly-shaped one — not the opposite.

**Cost:** L_pres costs 1.64 dB PSNR overall (95% CI [1.08, 2.22] dB, established, non-zero), consistent with the 1.46 dB seen on the ablation's own smaller eval set. Section 2 tests whether that cost buys anything in detection.

---

## 2. The Step 3 detection grid — what actually held up under scrutiny

**Question:** does preserving the defect residual (Section 1) translate into
better anomaly-detection AUROC — the only thing an inspection engineer
actually cares about? This is a link this project had never measured before
Step 3.

**Setup:** `src/experiments/eval_grid.py`, frozen PaDiM and PatchCore
(anomalib defaults, fit once per category on clean normals only, never
adapted to degraded/restored images), 5 restorers (identity, classical
wiener, restormer_deblur, and both L_pres-ablation checkpoints as
independently-selectable restorers `divide_lpres_on`/`_off`), 4 degradation
families (defocus, motion, noise, mixed), severities 2–4, 3 categories
(carpet, bottle, screw) — 360 cells, `results/step3_grid.csv` (5760 image
rows) / `step3_grid_pixel.csv` (360 cells). Image- and pixel-level AUROC
both computed (`anomaly_map` already returned by the detector at no extra
inference cost).

### 2.1 A result that needed gating before it meant anything

The grid's clean-image AUROC came back **PaDiM 0.867, PatchCore 0.781** —
PatchCore roughly 20 points below its ~0.99 published MVTec figure, before
any degradation was even applied. That gap needed explaining before any
downstream detection number could be trusted.

**Configuration check** (`src/experiments/patchcore_repro_check.py`):
PatchCore's actual constructor call in this project (`src/detect/harness.py`)
uses anomalib's own defaults — backbone `wide_resnet50_2`, layers
`layer2`/`layer3`, `coreset_sampling_ratio=0.1`, `num_neighbors=9` — which
match the published "PatchCore-10%" configuration exactly. The one thing
that does not match: the grid used `n_train=16`, `n_test=16` per category
(chosen for the speed of a 360-cell, 5-restorer grid), against full MVTec
splits of **209/280/320 train and 83/117/160 test** for bottle/carpet/screw
respectively — 13–20× fewer training images, 5–10× fewer test images.

**Reproduction check:** PatchCore refit and scored at the *published*
configuration — full train split, full test split, 256 resolution, clean
images only, no restoration, no degradation (`results/patchcore_repro_check.json`):

| category | AUROC | n_train | n_test |
|---|---|---|---|
| bottle | **1.000** | 209 | 83 |
| carpet | **0.919** | 280 | 117 |
| screw | **0.971** | 320 | 160 |
| **mean** | **0.963** | | |

This clears the harness-soundness bar. **The harness is not buggy** — the
grid's 0.781 clean PatchCore AUROC is a consequence of scoring at
`n_train=16` (a coreset built from a fraction of the normal patches a
10%-sampling memory bank needs to cover) and `n_test=16`, not a defect in
the detection code path. (Carpet's individual 0.919 sits a few points under
the informal "0.95+" bar even at full scale — noted, not chased; it does
not change the harness-soundness conclusion, since the *mean* and both other
categories clear it comfortably and none is remotely close to the grid's
20-point gap.)

**Consequence for reading the grid below:** all detection numbers in this
grid are measured at a **reduced-data configuration** (`n_train=16`,
`n_test=16`), not the published one. They are internally comparable —
every restorer and both detectors see the exact same reduced train/test
sets — but the *absolute* AUROC values are not representative of
production-scale PatchCore/PaDiM performance. The *relative* comparison
between restorers, at this same reduced scale, is what Section 2.2 tests.

### 2.2 Does restoration help detection? Only for some restorers, and only on one detector, with real support

Naive `gap_closed` (fraction of the clean→degraded AUROC gap recovered)
averaged per cell blew up on cells where `auroc_clean≈auroc_degraded`
(pixel-level cells hit −9 to −12) — the same small-denominator failure mode
already documented for relative DRR (`metrics/core.py`). Corrected with
ratio-of-means (`mean(method−degraded)/mean(clean−degraded)`, pooled —
`results/step3_grid_gapclosed_agg.csv`), then given a **95% bootstrap CI**
resampling which of the 16 test images are included per category (the same
resampled image set applied across every family/severity cell for that
category, since they reuse the identical physical images —
`src/experiments/bootstrap_grid_gapclosed.py`, `results/step3_grid_gapclosed_bootstrap.csv`,
2000 resamples):

![gap_closed forest plot](figures/step3_gapclosed_forest.png)

| restorer | detector | gap_closed | 95% CI | established? |
|---|---|---|---|---|
| wiener | PaDiM | −0.441 | [−0.727, −0.189] | **yes — negative** |
| divide_lpres_on | PaDiM | −0.277 | [−0.469, −0.130] | **yes — negative** |
| divide_lpres_off | PaDiM | −0.327 | [−0.656, −0.101] | **yes — negative** |
| restormer_deblur | PaDiM | +0.034 | [−0.043, 0.120] | no |
| wiener | PatchCore | +0.089 | [−0.001, 0.173] | no (borderline) |
| restormer_deblur | PatchCore | +0.044 | [−0.105, 0.141] | no |
| divide_lpres_on | PatchCore | +0.189 | [0.087, 0.285] | **yes — positive** |
| divide_lpres_off | PatchCore | +0.190 | [0.067, 0.309] | **yes — positive** |

**Read exactly this far and no further:**

- **DIVIDE (both L_pres variants) shows an established, opposite-signed
  effect by detector**: it measurably *hurts* PaDiM detection (both CIs
  exclude zero, negative) and measurably *helps* PatchCore detection (both
  CIs exclude zero, positive). This specific split is statistically
  supported, not just a point-estimate artifact.
- **`wiener` shows the same negative effect on PaDiM, established** — but
  its apparent PatchCore benefit is *not* established (CI [−0.001, 0.173]
  nearly touches zero). One-sided support, not a mirror of DIVIDE's pattern.
- **`restormer_deblur` shows no established effect on either detector** —
  both CIs straddle zero comfortably. At this grid's scale (n_test=16), we
  cannot distinguish restormer_deblur's effect on detection from no effect
  at all, in either direction.
- **A plausible mechanism** — PaDiM's per-patch Gaussian model penalizing
  any restoration artifact as distribution shift, versus PatchCore's
  nearest-neighbour memory bank tolerating small appearance changes more
  readily — is *consistent with* the DIVIDE and (one-sided) wiener results,
  but this grid does not test the mechanism directly and cannot confirm it;
  it is stated here as a hypothesis the data does not contradict, not a
  finding.
- **L_pres's own effect on detection is small and mostly a pixel-level
  effect, not an image-level one**: `divide_lpres_on` vs `_off` gap_closed
  is nearly identical at image level on PatchCore (0.189 vs 0.190) — the
  two checkpoints are statistically indistinguishable in whether an image
  gets flagged — but pixel-level point estimates diverge more (0.299 vs
  0.105, `results/step3_grid_gapclosed_agg.csv`; no CI computed at pixel
  level, see Limitations). L_pres's benefit, to the extent the grid can see
  it, looks like *where* the anomaly map lands, not *whether* the image is
  flagged — consistent with Section 1's residual-correlation (shape, not
  magnitude) framing.

**Bottom line for Section 2: "does preserving the defect residual help
detection" does not have a single yes/no answer at this grid's scale.**
It helps, with statistical support, specifically for DIVIDE on PatchCore. It
hurts, with statistical support, for DIVIDE and wiener on PaDiM. For
restormer_deblur, the grid is underpowered to say either way. This is not
"DIVIDE works" — it is "DIVIDE's detection effect is real but
detector-dependent," a materially narrower and more honest claim.

### 2.3 Companion measurement: scratch DRR and residual correlation at grid scope

`src/experiments/drr_study.py` run at the same scope as the grid (3
categories × 4 families × 3 severities, 8 base images/category,
`results/step3_drr_companion.csv`, 1440 rows) — the synthetic
counterfactual-pair measurement the grid's real-MVTec-image AUROC cannot
provide directly (DRR needs a paired clean/defect image degraded
identically; real MVTec test images don't have that pair).

![scratch DRR and residual correlation](figures/step3_scratch_drr_corr.png)

| restorer | scratch relative DRR | scratch residual corr | scratch PSNR (normal regions) |
|---|---|---|---|
| identity | 1.000 | 0.649 | 23.38 |
| wiener | 0.268 | 0.024 | 16.51 |
| restormer_deblur | 1.111 | 0.707 | 23.49 |
| divide_lpres_on | 1.189 | 0.778 | 25.64 |
| divide_lpres_off | 0.865 | 0.590 | 26.15 |

Consistent with Section 1's bootstrap finding at a broader scope (all 4
degradation families, not just the ablation's holdout mix): DIVIDE-with-
L_pres has the highest scratch residual correlation of any restorer tested,
including restormer_deblur. `divide_lpres_off`'s scratch DRR (0.865) sits
*below* identity (1.000) at this broader scope — without L_pres, DIVIDE
measurably erodes scratches here, reinforcing that L_pres is doing real,
non-trivial work rather than marginal polish.

---

## Figures

| file | shows |
|---|---|
| `figures/step3_gapclosed_forest.png` | gap_closed with 95% bootstrap CI, all 10 restorer×detector pairs |
| `figures/step3_auroc_by_restorer.png` | clean / degraded / restored image-AUROC bars, both detectors |
| `figures/step3_scratch_drr_corr.png` | scratch relative DRR + residual correlation by restorer, grid scope |
| `figures/drr_frontier_step3_drr_companion.png` | preservation-fidelity frontier, grid-scope companion study |
| `figures/drr_vs_severity_step3_drr_companion.png` | DRR vs. severity, grid-scope companion study |
| `figures/drr_by_kind_step3_drr_companion.png` | DRR by anomaly kind, grid-scope companion study |

## Result files

| file | contents |
|---|---|
| `results/ablate_lpres_bootstrap.json` | Section 1's bootstrap CIs |
| `results/step3_grid.csv` / `_pixel.csv` / `_summary.csv` | raw 360-cell grid, image and pixel level |
| `results/step3_grid_gapclosed_agg.csv` / `_by_family.csv` | ratio-of-means gap_closed, point estimates |
| `results/step3_grid_gapclosed_bootstrap.csv` | Section 2.2's bootstrap CIs |
| `results/patchcore_repro_check.json` | Section 2.1's published-config reproduction check |
| `results/step3_drr_companion.csv` / `_by_kind.csv` | Section 2.3's companion study |

## Limitations

- **The grid runs at a reduced data scale** (`n_train=16`, `n_test=16` per
  category vs. MVTec's full 83–320), chosen for a 360-cell × 5-restorer
  grid's runtime, not to match published numbers. Section 2.1's check
  confirms the harness is sound at full scale; it does not make the grid's
  own numbers full-scale numbers. A production-scale re-run (full
  train/test splits) has not been done and would very plausibly move both
  the absolute AUROCs and the gap_closed estimates.
- **No CI on pixel-level gap_closed.** Section 2.2's bootstrap covers
  image-level gap_closed only; the pixel-level point estimates reported in
  `step3_grid_gapclosed_agg.csv` (and referenced in 2.2's L_pres discussion)
  carry the same small-denominator risk this project has already documented
  and have not been resampled. Treat them as directional, not established.
- **`restormer_deblur`'s null result is "not established," not "no
  effect."** The bootstrap CI straddling zero on both detectors means this
  grid cannot distinguish a real small effect from none at n_test=16 — a
  larger grid could still find one.
- **The PaDiM/PatchCore mechanism (per-patch Gaussian vs. memory-bank
  tolerance) is a hypothesis consistent with the data, not something this
  grid tests directly.** No experiment here manipulates the detector's
  internal tolerance mechanism; confirming it would need a different study.
- **Correlational, not causal, throughout** — DRR, residual correlation, and
  gap_closed are all measured associations between a restorer and an
  outcome on a fixed set of synthetic/real degradations; no claim here is
  about what would happen under a different, untested degradation
  distribution.
- Every other limitation in `README.md`'s own Limitations section
  (illumination-correction ineffectiveness at high severity, the
  `bottle`/`defocus`/severity-3 kernel misclassification, etc.) still
  applies and is not repeated here.
