# DIVIDE

**D**egradation-**I**nvariant **V**isual **I**nspection via **D**ecomposition and **E**nhancement

Digital Image Processing (BCSE403L) course project. Atharv Agarwal, VIT Vellore.

---

## The claim

Industrial inspection pipelines run `capture → restore → detect`. This project
started from a broad argument: that step 2 is self-defeating in general —
that a restoration model is a denoiser, a defect is statistically a sparse
local anomaly, so any restorer suppresses the evidence the detector needs.

**The go/no-go study (`results/drr_study_gonogo*`) tested that broad claim and
did not support it.** A real, competently-functioning learned restorer —
Restormer's `real_denoising` checkpoint, not a broken or out-of-distribution
one — does genuine restoration (+2.3dB PSNR over doing nothing) while
preserving defect signal (relative DRR 0.918 mean, 0.881 on scratches
specifically). That is direct evidence against "learned natural-image priors
erase sparse defects," not a gap in the test.

What the same data shows instead is a clean split by **mechanism**, not by
learned-vs-classical: restorers that deconvolve — invert a blur kernel —
erode scratches severely (Wiener 0.335, a classical deconvolution pipeline
0.516 relative DRR); restorers that merely denoise — attenuate high-frequency
energy without inverting anything — do not (Restormer, NLM, Gaussian,
bilateral: 0.88–0.98). The physical reason: deconvolution is an ill-posed
inverse problem, so a deconvolving restorer necessarily uses a prior to
decide what the sharp image *should* have looked like, and a sparse defect
looks exactly like the thing that prior is trained to remove. Denoising has
no such inversion step — it attenuates energy, and a defect with real
contrast against its surroundings survives attenuation by construction.

**The narrowed, supported claim: deblurring-class restoration erases sparse
defects, because deconvolution is an ill-posed inverse that relies on a prior
to decide what should have been there. DIVIDE performs deblurring in a way
that structurally cannot do this**, because its inverse operator (the
closed-form Wiener data step in PCIM's HQS recursion — no learned parameters)
is content-agnostic: `tests/test_pcim.py::test_hqs_without_prox_is_linear`
verifies this directly, showing the deconvolution path is provably an affine
map of the input with the learned component gated off. DIVIDE was always a
deconvolution method — the Wiener step is its core — so this is a narrower
and sharper version of the original bet, not a retreat from it: it's
supported by the most dramatic number this project has produced (0.335
relative DRR for classical Wiener on scratches), it explains *why* naive
restore-then-detect specifically fails in industrial settings, where motion
blur and defocus are dominant degradations, and it says plainly where the
broad claim does and doesn't hold rather than asserting it holds everywhere.

A follow-up run (`results/drr_study_deblur*`, using Restormer's own
Motion_Deblurring/Defocus_Deblurring checkpoints rather than its denoising
one) is testing whether a *learned* deblurrer erodes defects the way
classical Wiener does, or whether Wiener's erosion is instead a
regularization artifact specific to the classical method. Both answers are
reportable and neither changes the narrowed claim above, which rests on the
classical result and DIVIDE's own structural argument either way — see
`HANDOFF.md` for the running result once it lands.

---

## Status

| Component | State |
|---|---|
| Degradation simulator (5 families × 5 severities + mixed) | done, tested |
| Synthetic anomaly generator (texture / scratch / blob) | done, tested |
| Metrics: DRR, relative DRR, ACG, HDR, DRemR, AUROC, AP, F1, gap_closed | done, tested |
| DBDE — defect-blind degradation estimator | done, tested, **validated on real MVTec** — see `figures/dbde_*.png` |
| DRR go/no-go study + figures | **done on real MVTec, CPU, including NAFNet/Restormer** (`results/drr_study_gonogo*.csv`) — see **The claim** above for the verdict and its two caveats; classical-only baseline also on disk (`results/drr_study_real*.csv`) |
| Frozen-detector harness (PatchCore/PaDiM/ReverseDistillation/EfficientAd) | built, PaDiM verified on CPU; others fit+scored at least once but flaky in dev sandbox — **never run on real MVTec or a GPU** |
| Deep restorer loaders (NAFNet/Restormer/DiffBIR) | NAFNet/Restormer run in-process on CPU, no GPU needed (weights downloaded, ~2–7s/image) — see `src/models/deep_restorers.py`; **DiffBIR still needs a GPU**, excluded from CPU studies on compute grounds |
| PCIM — physics-consistent inverse module | built, tested (incl. the structural content-agnostic claim); **trained on CPU** (`configs/train_pcim_cpu.yaml`, 15000 steps) — see `results/train_pcim_cpu_eval.csv` |
| Losses — L_rec / L_deg / L_freq / L_pres | built, tested against the real counterfactual pipeline |
| SARG — sparse anomaly-residual guard | built, tested; **never trained** |
| PCIM training script (resumable, CSV-logged) | built, tested incl. a real kill-and-resume check; **run to completion on CPU** — GPU run still unmeasured |
| L_pres ablation harness | built, tested (incl. the fairness-assertion that both runs differ ONLY in `use_lpres`); **never run at meaningful scale** |
| Frozen-detector evaluation grid (resumable, CSV) | built, verified end-to-end on synthetic data; **never run on real MVTec** |
| Gradio demo | built, panel logic + Blocks construction verified; **never run against real weights** |

244+ unit tests, all passing, all CPU. **Read [HANDOFF.md](HANDOFF.md) before running anything on a GPU** — it has the exact run order, every weight URL, and what "verified" does and doesn't mean for each piece above.

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

## The go/no-go experiment

**Run this before building anything else.** It decides whether the project is
viable, and it costs two days instead of a semester.

```bash
python -m src.experiments.drr_study --categories carpet bottle screw
```

It pastes a synthetic defect with a known mask, degrades the defect version and
the defect-free version identically, restores both, and measures how much of the
defect residual survives.

Read `results/drr_study_summary.csv` and `figures/drr_frontier.png`:

| relative DRR | meaning |
|---|---|
| < 0.5 | restorers erase defects → **GO**, build DIVIDE |
| 0.5–0.8 | ambiguous → inspect per-restorer and scratch-only tables |
| > 0.8 | restorers preserve defects → **NO-GO**, pivot (see Plan B in the spec) |

Judge on **relative** DRR, not raw. Raw DRR also counts attenuation caused by
the degradation itself, which no restorer is responsible for; the identity
baseline scores ~0.87 raw and exactly 1.0 relative.

**The classical restorers alone will not settle this.** The verdict is only
meaningful once at least one competently-functioning learned restorer is in
the run — NAFNet and Restormer both run in-process on CPU now (see
`src/models/deep_restorers.py`; no GPU required, ~2–7s/image), DiffBIR is
excluded on compute grounds (a full diffusion sampler, no realistic CPU path
at this scope — see its `REPO_SPECS` entry for the reasoning).

**Result** (`results/drr_study_gonogo*`, reduced scope: 3 categories, 3
families, severities 2–4, 10 images, 2700 rows): raw pooled verdict
NO-GO, but two of the nine non-identity restorers distort that number —
msrcr/clahe are tone-mapping methods whose DRR > 1 reflects amplification
artefacts, not preservation, and NAFNet-SIDD was out-of-distribution on
these synthetic degradations (its own PSNR came back *worse* than the raw
degraded input, 21.11 vs 22.84dB — a broken baseline, not evidence).
Excluding both, the real result is a clean mechanism split, not a
learned-vs-classical one — see **The claim** above for the full narrowed
argument and the follow-up test (`results/drr_study_deblur*`) checking
whether a learned deblurrer behaves like classical Wiener.

---

## Running the GPU work

Everything above is CPU. These need a GPU:

| Job | Cost | Notes |
|---|---|---|
| Deep restorers in the DRR study | ~1–2 GPU-h | needs weights (Google Drive for NAFNet/Restormer, manual download — see HANDOFF.md); loaders never executed |
| Frozen-detector harness on real MVTec | ~1 GPU-h | PaDiM verified on CPU/synthetic only; PatchCore/ReverseDistillation/EfficientAd built but never run on a GPU or real data |
| PCIM + SARG training | not estimable yet | **training script doesn't exist** — see HANDOFF.md step 5 |
| Full evaluation grid | ~10–15 GPU-h, floor not ceiling | resumable (`eval_grid.py`), embarrassingly parallel across cells but not yet parallelized in code |

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
configs/                 experiment configs (yaml)
scripts/smoke.sh         full CPU verification, run before every push
src/utils/paths.py       environment-aware paths — never hardcode a path
src/degrade/simulator.py degradation families, severities, counterfactual pairs
src/degrade/anomaly.py   synthetic anomaly generation
src/dbde/estimator.py    defect-blind degradation estimator (pure DIP, no GPU)
src/metrics/core.py      DRR / ACG / HDR / DRemR / AUROC — unit-tested
src/models/restorers.py  classical restorer registry + get_restorer()/available_restorers()
src/models/deep_restorers.py  NAFNet/Restormer/DiffBIR loaders — clone repo, fail loud if weights missing
src/models/divide_restorer.py DIVIDE itself as a Restorer (DBDE -> PCIM -> optional SARG)
src/models/pcim.py       Physics-Consistent Inverse Module (PyTorch)
src/models/losses.py     L_rec, L_deg, L_freq, L_pres
src/models/sarg.py       Sparse Anomaly-Residual Guard (unrolled RPCA)
src/detect/harness.py    frozen-detector harness (anomalib: PatchCore/PaDiM/ReverseDistillation/EfficientAd)
src/detect/_compat.py    runtime shims anomalib needs on a current stack — read before touching anomalib imports
src/data/mvtec.py        MVTec loading, with synthetic smoke fallback
src/demo/app.py          Gradio demo — four panels, live relative-DRR readout
src/experiments/drr_study.py       go/no-go experiment
src/experiments/eval_grid.py       full frozen-detector evaluation grid
src/experiments/train_pcim.py      PCIM training loop — config-driven, resumable
src/experiments/ablate_lpres.py    trains WITH/WITHOUT L_pres, reports the relative-DRR delta
src/experiments/dbde_validation.py defect-blindness / parameter-accuracy / reference-vs-blind figures
configs/train_pcim_cpu.yaml        128x128 overnight-CPU-run config
configs/train_pcim_gpu.yaml        256x256 GPU config
tests/                   225 tests, CPU only
HANDOFF.md               read this before running anything on a GPU
```

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
