# DIVIDE

**D**egradation-**I**nvariant **V**isual **I**nspection via **D**ecomposition and **E**nhancement

Digital Image Processing (BCSE403L) course project. Atharv Agarwal, VIT Vellore.

**[FINDINGS.md](FINDINGS.md)** is the full report: the mechanism, the L_pres
result, and the detection-grid result, each at the confidence level its own
evidence supports. This README is orientation and status; FINDINGS.md is
the numbers.

---

## The claim

This project is structured as an investigation, not a pitch, so this section
reads as one: the hypothesis it started with, how it was tested, what got
falsified, and what actually held up. All numbers below are **ratio-of-means**
relative DRR (see `src/metrics/core.py`'s `relative_drr` docstring for why
that specific convention, and not the more obvious per-example-average one) —
every table this project reports uses this same definition.

**Original hypothesis.** Industrial inspection pipelines run
`capture → restore → detect`. Restoration is self-defeating for this purpose:
a restoration model is fundamentally a denoiser, a surface defect is
statistically a sparse local anomaly, so any restorer suppresses the evidence
the detector needs. DIVIDE's answer was to invert the physical degradation
instead of denoising, because an inverse operator can be made
content-agnostic in a way a learned denoiser structurally cannot.

**Tested against learned denoisers — not supported.** Restormer's
`real_denoising` checkpoint (`results/drr_study_gonogo*`), a real,
competently-functioning learned restorer, does genuine restoration (+2.3dB
PSNR over doing nothing) while preserving defect signal (scratch relative DRR
0.867). That's direct evidence against "learned natural-image priors erase
sparse defects," not a gap in the test. NLM, Gaussian, and bilateral —
classical denoisers — land in the same 0.85–0.98 band. Denoising, learned or
classical, does not show the failure mode the hypothesis predicted.

**Tested against learned deblurrers — also not supported.** Restormer's own
Motion_Deblurring/Defocus_Deblurring checkpoints (`results/drr_study_deblur_corr*`,
dispatched per image by DBDE's blur-kind estimate), not its denoising one,
preserve too: scratch relative DRR 1.22, and — the check that mattered, see
below — residual correlation 0.70, well above the un-restored baseline's own
0.59. A learned model actually trained to deblur does not erode defects
either.

**The one class that does erode: single-shot closed-form Wiener
deconvolution.** Classical Wiener (`results/wiener_bridge_ablation*`) gives
scratch relative DRR 0.079; a `classical_pipeline` chaining illumination
correction, denoising, and Wiener deblurring gives 0.571
(`results/drr_study_gonogo*`). Swept across five orders of magnitude of its
own regularization constant (`results/wiener_nsr_sweep*`,
`figures/wiener_nsr_sweep.png`), scratch **residual correlation** — not just
DRR — stays at or near zero throughout (-0.06 to +0.12, all within noise of
zero, nsr from 1e-5 to 1.0). The defect isn't attenuated by some regularization
settings and preserved by others; it's gone at every setting tested. This is
the most dramatic, and best-supported, single result this project has
produced.

**The mechanism, located by a bridging ablation.** Five configurations
walking from classical Wiener to PCIM's `x_cons`, one variable at a time
(`results/wiener_bridge_ablation*`):

| step | change | scratch relative DRR | scratch residual corr |
|---|---|---|---|
| A | classical Wiener, flat nsr | 0.079 | -0.058 (erosion, confirmed real) |
| B | + nsr derived from DBDE's own estimate | 0.129 | -0.053 (same) |
| C | + unrolled HQS (5 iters, growing ρ), still raw-pixel domain | 1.377 | 0.864 (flips to genuine preservation) |
| D | + VST/illumination domain (= `x_cons`) | 1.379 | 0.859 |
| E | `x_cons`, regularization floor removed | 1.319 | 0.563 (back near baseline, not erosion) |

**Iteration, not regularization, is decisive.** At matched regularization
magnitude, switching one-shot division (B) for unrolled half-quadratic
splitting (C) flips scratch residual correlation from -0.05 to 0.86 — the
defect goes from gone to genuinely preserved. Candidate mechanism: each HQS
step solves a data term anchored toward the *previous* iterate, so even at
negligible regularization the recursion takes several partial steps rather
than committing in one shot to the full inversion that destroys sparse
content. Regularization has a separate, smaller job: config E shows that
removing PCIM's `nsr_floor` keeps the safety property (no erosion — 0.563 is
nowhere near A/B's near-zero) but loses the *gain* — E's correlation drops
back to roughly the do-nothing baseline (0.59), while C/D exceed it (0.86).
Iteration is what prevents the failure; adequate regularization is what
turns "no worse than doing nothing" into "measurably better."

**What DIVIDE therefore demonstrates.** Its unrolled HQS formulation is
exactly the property that avoids single-shot Wiener's erosion — the
architecture was correct, but the original explanation for *why* was not:
"content-agnostic" was the right instinct, but the failure mode it prevents
turned out to be specific to one-shot inversion, not to restoration or even
to deconvolution in general. DIVIDE does not solve a problem all restoration
methods have — competent learned denoisers and deblurrers largely don't have
it. What it demonstrates is narrower and still real: the specific failure of
single-shot closed-form inversion is avoidable, DIVIDE's architecture avoids
it structurally rather than by luck (`tests/test_pcim.py::test_hqs_without_prox_is_linear`
verifies the gated-off path is provably an affine, content-agnostic map), and
that failure mode is exactly the one industrial inspection is most exposed
to, since motion blur and defocus — the degradations a naive Wiener step
would be reached for — dominate real inspection settings.

**Two findings about this project's own metrics,** surfaced by stress-testing
the claim above rather than by design:

- **DRemR is biased at small degradation magnitude** (`degradation_removal_ratio`'s
  denominator) — it reads "made it worse" on mildly-degraded images that were
  in fact restored well, and "did nothing" on severely-degraded images that
  were in fact restored badly. Report it alongside absolute PSNR, never alone.
- **DRR alone cannot distinguish genuine preservation from ringing once it
  approaches or exceeds ~1.0** — a restorer can leave residual energy at a
  defect's location with the right magnitude but the wrong shape. It needs
  `defect_residual_correlation` (Pearson correlation between the restored and
  true residual, mean-centred within the mask) as a companion measurement.
  Below relative DRR ~0.3–0.4 the two metrics always agreed throughout this
  project's data, so the erosion boundary above was never actually in doubt —
  only the readings at or above ~1.0 needed the second check, and every one
  checked out as genuine except config E, which the correlation check
  correctly demoted from "as good as C/D" to "no better than doing nothing."

**Two results build directly on the mechanism above and are reported in
full in [FINDINGS.md](FINDINGS.md), not duplicated here:**

- **L_pres — established.** A 150-example held-out bootstrap comparison of
  PCIM trained with vs. without the preservation loss: scratch residual-
  correlation delta 95% CI [0.188, 0.296], excludes zero. Given the
  iteration mechanism above already prevents erasure, L_pres still adds a
  real, measurable amount of shape fidelity on top of it, at a measured
  cost of 1.64 dB PSNR.
- **Does preserving the residual help detection?** A 360-cell frozen-
  detector grid (PaDiM + PatchCore, 5 restorers, real MVTec), gated by a
  harness-correctness check (PatchCore reproduces its published AUROC —
  0.963 mean vs. published ~0.99 — at full train/test scale; the grid
  itself runs at a reduced 16-image-per-category scale for speed) and a
  bootstrap CI on every headline number. Result: **yes for DIVIDE on
  PatchCore, no for DIVIDE on PaDiM, both statistically established** — a
  real, detector-dependent effect, not a uniform "restoration helps"
  finding, and specific to the few-normal-training regime the grid
  actually tests (untested at published training scale).

---

## Status

| Component | State |
|---|---|
| Degradation simulator (5 families × 5 severities + mixed) | done, tested |
| Synthetic anomaly generator (texture / scratch / blob) | done, tested |
| Metrics: DRR, relative DRR, ACG, HDR, DRemR, AUROC, AP, F1, gap_closed | done, tested |
| DBDE — defect-blind degradation estimator | done, tested, **validated on real MVTec** — see `figures/dbde_*.png` |
| DRR go/no-go study + figures | **done on real MVTec, CPU, including NAFNet/Restormer** (`results/drr_study_gonogo*.csv`) — see **The claim** above; classical-only baseline also on disk (`results/drr_study_real*.csv`) |
| Learned-deblurring test | **done on real MVTec, CPU** (`results/drr_study_deblur_corr*.csv`) — Restormer's own deblurring checkpoints, not the denoising one |
| Bridging ablation + nsr sweep (mechanism) | **done on real MVTec, CPU** (`results/wiener_bridge_ablation*.csv`, `results/wiener_nsr_sweep*.csv`, `figures/wiener_nsr_sweep.png`) — locates the iteration-vs-regularization mechanism, see **The claim** |
| `defect_residual_correlation` (shape check alongside DRR) | **done, tested, applied retroactively to every table above** — see **The claim**'s metrics findings |
| Frozen-detector harness (PatchCore/PaDiM/ReverseDistillation/EfficientAd) | **PaDiM and PatchCore run on real MVTec, CPU** — PatchCore verified against its published AUROC at full train/test scale (`results/patchcore_repro_check.json`, mean 0.963) and both run through the full detection grid (`FINDINGS.md` §3); ReverseDistillation/EfficientAd still only fit+scored once each, flaky in dev sandbox — **never run on real MVTec or a GPU** |
| Deep restorer loaders (NAFNet/Restormer/DiffBIR) | NAFNet/Restormer + Restormer's deblurring checkpoints run in-process on CPU, no GPU needed (weights downloaded, ~2–7s/image) — see `src/models/deep_restorers.py`; **DiffBIR still needs a GPU**, excluded from CPU studies on compute grounds |
| PCIM — physics-consistent inverse module | built, tested (incl. the structural content-agnostic claim); **trained on CPU** (`configs/train_pcim_cpu.yaml`, 15000 steps) — see `results/train_pcim_cpu_eval.csv` |
| Losses — L_rec / L_deg / L_freq / L_pres | built, tested against the real counterfactual pipeline |
| SARG — sparse anomaly-residual guard | built, tested; **never trained** |
| PCIM training script (resumable, CSV-logged) | built, tested incl. a real kill-and-resume check; **run to completion on CPU** — GPU run still unmeasured |
| L_pres ablation harness | built, tested (incl. the fairness-assertion that both runs differ ONLY in `use_lpres`); **trained to completion on CPU and re-evaluated on the full 150-example held-out set with bootstrap CIs — the central open question is answered: established, see `FINDINGS.md` §2** |
| Frozen-detector evaluation grid (resumable, CSV) | **run on real MVTec, CPU** — 360 cells, PaDiM + PatchCore × 5 restorers × 4 families × severities 2–4 × 3 categories, gated by a harness-reproduction check and bootstrapped `gap_closed` CIs — see `FINDINGS.md` §3. Runs at a reduced 16-image-per-category scale (untested at published training scale — see `FINDINGS.md` §3.2's data-scarce framing) |
| Gradio demo | **rebuilt as precomputed-lookup-only** (`src/demo/precompute.py` generates 21 real-MVTec examples against real weights - wiener/restormer_deblur/`divide_lpres_on` - once; `src/demo/app.py`/`static_demo.py` only read that manifest, no live inference) - see `demo.html` and Layout below |

300+ unit tests, all passing, all CPU. **Read [HANDOFF.md](HANDOFF.md) before running anything on a GPU** — it has the exact run order, every weight URL, and what "verified" does and doesn't mean for each piece above.

---

## Quick start

```bash
pip install -r requirements.txt
./scripts/smoke.sh            # full pipeline, no dataset, no GPU, ~30 s
```

If that prints `ALL GREEN`, the code is working. Then with real data:

```bash
export DIVIDE_DATA_ROOT=/path/to/parent/of/mvtec_ad
python -m src.experiments.drr_study --categories carpet bottle screw
```

---

## Data

Neither dataset can be downloaded automatically — both need a human.

| Dataset | Where | Note |
|---|---|---|
| MVTec AD | https://www.mvtec.com/company/research/datasets/mvtec-ad | licence form, ~5 GB |
| DTD | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | ~600 MB, source textures for synthetic anomalies |

Expected layout:

```
$DIVIDE_DATA_ROOT/
  mvtec_ad/bottle/train/good/000.png
  dtd/images/banded/banded_0002.jpg
```

Paths resolve automatically on Kaggle (`/kaggle/input/...`). Nothing is
hardcoded — see `src/utils/paths.py`. Upload both as **private** Kaggle
datasets; the MVTec licence forbids redistribution.

---

## The go/no-go experiment, and what came after it

```bash
python -m src.experiments.drr_study --categories carpet bottle screw
```

Pastes a synthetic defect with a known mask, degrades the defect version and
the defect-free version identically, restores both, measures how much of the
defect residual survives (`results/<tag>_summary.csv`,
`figures/drr_frontier_<tag>.png`; `--tag` defaults to `drr_study`, figures are
tag-suffixed so different studies never overwrite each other's — see
`src/experiments/drr_study.py`).

| relative DRR | meaning |
|---|---|
| < 0.5 | restorers erase defects |
| 0.5–0.8 | ambiguous — inspect per-restorer, per-kind, and (if it approaches or exceeds ~1.0) residual-correlation tables |
| > 0.8 | restorers preserve defects |
| ≥ ~1.0 | **check `defect_residual_correlation` before reading this as preservation** — see **The claim** |

Judge on **relative** DRR (ratio-of-means, `src/metrics/core.py`), not raw —
raw DRR also counts attenuation caused by the degradation itself, which no
restorer is responsible for.

The actual sequence run, each building on the last:

1. **Go/no-go** (`results/drr_study_gonogo*`, 3 categories, 3 families,
   severities 2–4, 10 images, 2700 rows, including NAFNet and Restormer's
   denoising checkpoint) — raw pooled verdict NO-GO, but msrcr/clahe
   (tone-mappers, not restorers — DRR > 1 there is amplification, not
   preservation) and NAFNet (out-of-distribution on these degradations —
   its own PSNR came back *worse* than the raw input) distort the pooled
   number. Excluding both: a mechanism split, not a learned-vs-classical one.
2. **Learned-deblurring test** (`results/drr_study_deblur_corr*`) — the
   mechanism split held for denoisers; did it hold for deblurrers too?
   Restormer's actual deblurring checkpoints, not the denoising one: yes.
3. **Bridging ablation + nsr sweep** (`results/wiener_bridge_ablation*`,
   `results/wiener_nsr_sweep*`) — located *why*: iteration, not
   regularization strength, separates erosion from preservation.

See **The claim** above for the full numbers and reasoning at each step.

---

## Running the GPU work

Everything above is CPU. These need a GPU:

| Job | Cost | Notes |
|---|---|---|
| PCIM to real convergence, SARG training | not estimable yet | PCIM ran to completion on CPU (`configs/train_pcim_cpu.yaml`, see **The claim**/Status) but not at GPU scale/resolution; SARG has never been trained at all |
| Evaluation grid at published train/test scale | ~10–15 GPU-h, floor not ceiling on CPU; faster on GPU | The 360-cell grid already ran on CPU, real MVTec, at a reduced `n_train=16`/`n_test=16` scale (`FINDINGS.md` §3) — the harness itself is verified sound (`FINDINGS.md` §3.1's PatchCore reproduction check, full train/test scale, clean images only). What's untested is the *detection comparison* (not just the harness) at published train/test scale (200–320 train, 80–160 test) — resumable (`eval_grid.py`), embarrassingly parallel across cells but not yet parallelized in code |
| DiffBIR | seconds/image on a GPU vs. no realistic CPU path | needs weights (Hugging Face, curl/wget-able — see HANDOFF.md); loader never executed |

**Before any GPU session, run `./scripts/smoke.sh` and confirm `ALL GREEN`.**
Debugging on borrowed hardware while someone waits is the worst way to spend
that favour. **Then read [HANDOFF.md](HANDOFF.md)** — it has the actual run
order, the go/no-go stop, and exactly what's verified vs. never run.

### Kaggle

New notebook → GPU T4 ×2 → attach both datasets → one cell:

```python
!git clone -q https://github.com/USER/divide-dip.git /kaggle/working/divide
%cd /kaggle/working/divide
!pip install -q -r requirements.txt
!python -m src.experiments.drr_study --categories carpet bottle screw

# push results back - /kaggle/working is wiped when the session ends
!git config user.email "you@example.com" && git config user.name "you"
!git add results/ figures/ && git commit -qm "drr study" && git push -q
```

Use **Save Version → Save & Run All (Commit)** for long runs; interactive
sessions die on idle.

---

## Layout

```
FINDINGS.md              full report: mechanism, L_pres, detection grid, each at its earned confidence level
demo.html                static demo fallback (no server) — generated by src/demo/static_demo.py
demo_data/                21 precomputed examples (images + manifest.json) — generated by src/demo/precompute.py
configs/                 experiment configs (yaml)
scripts/smoke.sh         full CPU verification, run before every push
src/utils/paths.py       environment-aware paths — never hardcode a path
src/degrade/simulator.py degradation families, severities, counterfactual pairs
src/degrade/anomaly.py   synthetic anomaly generation
src/dbde/estimator.py    defect-blind degradation estimator (pure DIP, no GPU)
src/metrics/core.py      DRR / ACG / HDR / DRemR / AUROC — unit-tested
src/models/restorers.py  classical restorer registry + get_restorer()/available_restorers()
src/models/deep_restorers.py  NAFNet/Restormer (denoising + Motion/Defocus deblurring)/DiffBIR loaders
src/models/wiener_ablation.py      classical Wiener -> PCIM x_cons bridging ablation (5 configs)
src/models/divide_restorer.py DIVIDE itself as a Restorer (DBDE -> PCIM -> optional SARG)
src/models/pcim.py       Physics-Consistent Inverse Module (PyTorch)
src/models/losses.py     L_rec, L_deg, L_freq, L_pres
src/models/sarg.py       Sparse Anomaly-Residual Guard (unrolled RPCA)
src/detect/harness.py    frozen-detector harness (anomalib: PatchCore/PaDiM/ReverseDistillation/EfficientAd)
src/detect/_compat.py    runtime shims anomalib needs on a current stack — read before touching anomalib imports
src/data/mvtec.py        MVTec loading, with synthetic smoke fallback
src/demo/precompute.py  generates demo_data/ once (21 real-MVTec examples, real weights) - the only piece that runs a model
src/demo/app.py          Gradio demo — precomputed lookups only, degraded/wiener/restormer_deblur/DIVIDE x 4 panels
src/demo/static_demo.py  writes demo.html — same precomputed content, no server, images embedded as base64
src/experiments/drr_study.py       go/no-go experiment, resumable, ratio-of-means relative DRR
src/experiments/wiener_regularization_sweep.py  classical Wiener's nsr swept 1e-5 to 1.0
src/experiments/eval_grid.py       full frozen-detector evaluation grid (image + pixel AUROC, gap_closed)
src/experiments/bootstrap_grid_gapclosed.py  bootstrap CI on eval_grid's gap_closed, resampling test images
src/experiments/grid_figures.py    figures for the detection grid (forest plot, AUROC bars, scratch DRR/corr)
src/experiments/patchcore_repro_check.py  PatchCore at published config — harness-correctness gate for eval_grid
src/experiments/train_pcim.py      PCIM training loop — config-driven, resumable
src/experiments/ablate_lpres.py    trains WITH/WITHOUT L_pres, reports the relative-DRR delta
src/experiments/eval_pcim_holdout.py  post-hoc diagnostics on a trained PCIM checkpoint
src/experiments/compare_lpres_bootstrap.py  bootstrap CI on the L_pres ON/OFF delta, full held-out set
src/experiments/dbde_validation.py defect-blindness / parameter-accuracy / reference-vs-blind figures
configs/train_pcim_cpu.yaml        128x128 overnight-CPU-run config
configs/train_pcim_gpu.yaml        256x256 GPU config
tests/                   300+ tests, CPU only
HANDOFF.md               read this before running anything on a GPU
```

---

## Limitations

Stated plainly, not buried in a caveat clause:

- **No diffusion restorer measured** (DiffBIR) — compute, not a technical
  blocker; excluded from every CPU study, loader kept working for a GPU box.
- **Two detectors verified on real MVTec, CPU** (PaDiM and PatchCore —
  PatchCore's own published-AUROC reproduction check, `FINDINGS.md` §3.1,
  confirms the harness is sound) — ReverseDistillation and EfficientAd are
  built but never run on real MVTec or a GPU.
- **The detection grid's own headline result is scoped to a reduced-data
  regime** (`n_train=16`/`n_test=16` per category) and has not been re-run
  at published training scale (200–320 train images) — see `FINDINGS.md`
  §3.2. The harness reproduces published numbers at full scale; the
  *detection comparison* has not been re-run at that scale, and could move
  or vanish there.
- **HDR (hallucinated defect rate) is defined and tested, never measured**
  on a trained detector + restorer pair.
- **Every real-data DRR study so far uses a reduced grid** — 3 categories, a
  subset of families/severities, 8–10 images — not the full simulator's 6
  families × 5 severities. Directional, not exhaustive.
- **Illumination correction is ineffective at high severity** — `x_cons`
  PSNR tracks the raw degraded input's PSNR almost exactly at illumination
  severity 3–5 (within ~0.3–0.8dB), suggesting the illumination-divide step
  contributes little there. Not investigated further.
- **The illumination blur false-positive is not cleared, only not
  observed** — DBDE's reference-mode blur detector had a real false-positive
  rate (10–65%) on severely under-exposed images in isolated validation. It
  did not fire in either 150-example holdout draw used this session, but
  n was only 2–4 per severity there — absence of evidence, not evidence of
  absence.
- **`bottle`/`defocus`/severity-3 still misclassifies as motion** (oversized
  ~19–20px kernel) — same failure shape as the `carpet`/`defocus`/severity-1
  bug that was found and fixed, a different cell, not chased.
- **The -0.05 motion-vs-defocus margin is empirically calibrated, not
  derived** — a positive margin looked more principled but cost real
  true-positive rate (90% → 71% at motion severity 2) without moving the
  false-positive case it exists to fix. An honest tradeoff, not a clean
  constant.

---

## Two bugs worth knowing about

Both were caught by tests and would have silently corrupted results.

**1. Content-dependent noise broke the counterfactual pairs.** Drawing Poisson
counts directly consumes a different number of random values per pixel
depending on intensity, so two images differing only in a small defect region
received *different noise everywhere*. Every counterfactual pair — and therefore
`L_pres` itself — would have been invalid, with no visible symptom beyond a
training curve that never converged. Fixed by drawing standard normals first
and scaling by `sqrt(signal)` afterwards. See `add_poisson_gaussian`.

**2. Low-pass filtering before the illumination fit absorbed defects.** A
Gaussian pre-blur smears a scratch into a broad smooth bump that the robust
weights can no longer distinguish from genuine illumination, so the defect ends
up inside the estimated field `L` — exactly the failure the method exists to
prevent. Fixed by fitting the raw log image with Huber weights and a small
median prefilter. See `estimate_illumination`.

---

## Design notes

- **DBDE uses no learned weights and no GPU.** Its defect-blindness is verified
  directly in `tests/test_dbde.py::test_estimator_is_defect_blind`, which is the
  structural claim the whole method rests on.
- **Blur estimation has a reference mode.** Industrial inspection always has
  clean normals of the exact part, so `reference_psd()` over the training set
  turns blind blur estimation into a much easier reference-based problem. This
  is a real advantage over generic blind restoration and is worth stating in the
  report.
- **Global exposure is not recoverable from one image.** Scene albedo and
  illumination gain are not separable, so `correct_illumination` fixes only the
  spatial field unless you pass `match_mean` from the clean training normals.
- Everything is seed-fixed and config-driven. Results go to CSV keyed by
  `(restorer, category, family, severity, anomaly_kind)` so the grid assembles
  itself.
