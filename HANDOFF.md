# Handoff: DIVIDE, laptop → GPU machine

You have a GPU and (presumably) no context on this project beyond this document. Read it before running anything. It tells you exactly what to run, in what order, what a correct result looks like, and where the code is likely to be wrong because I wrote it without being able to execute it.

## What this project claims

**This section was rewritten after the full investigation finished (four CPU
studies, no GPU needed for any of them) — see README.md's "The claim" for
every number and the full reasoning. This is the short version; don't build
on it without reading the README section at least once.**

**Original hypothesis.** Restoration is self-defeating for industrial
inspection: any restorer, classical or deep, is fundamentally a denoiser, a
surface defect is a sparse anomaly, so a defect *is* noise to a model trying
to reconstruct a clean instance of its class, and it gets erased with the
genuine degradation. DIVIDE's answer was to invert the physical degradation
instead, because an inverse operator can be content-agnostic in a way a
learned denoiser structurally cannot.

**Tested three ways, the first two falsified it:**

1. Learned denoiser (Restormer `real_denoising`) — preserves defects
   (scratch relative DRR 0.867, genuine restoration alongside it, +2.3dB
   PSNR). Not supported.
2. Learned deblurrer (Restormer's own Motion/Defocus_Deblurring checkpoints,
   not the denoising one) — also preserves (scratch relative DRR 1.22,
   residual correlation 0.70, above the do-nothing baseline's own 0.59). Not
   supported either.
3. Classical single-shot Wiener deconvolution — this is the one that erodes.
   Scratch relative DRR 0.08–0.57 depending on scope, residual correlation
   pinned at/near zero across five orders of magnitude of its own
   regularization constant. Confirmed, not a fluke: this is the most
   dramatic and best-supported number this project has produced.

**The mechanism** (a bridging ablation, classical Wiener → PCIM's `x_cons`,
one variable at a time): switching one-shot division for PCIM's unrolled
half-quadratic-splitting recursion, at matched regularization strength,
flips scratch residual correlation from -0.05 to 0.86. Regularization
strength turned out NOT to be the mechanism (the reverse-direction check -
disabling PCIM's `nsr_floor` - did not bring erosion back); iteration is.
Candidate reason: each HQS step anchors toward the previous iterate rather
than committing to the full inversion in one shot.

**What DIVIDE actually demonstrates:** its unrolled HQS formulation is
exactly the property that avoids single-shot Wiener's erosion -
`tests/test_pcim.py::test_hqs_without_prox_is_linear` verifies the
gated-off path is provably affine and content-agnostic. The architecture was
right; the original justification ("restoration in general erases defects")
was not. DIVIDE does not solve a problem all restoration methods have -
competent denoisers and deblurrers mostly don't have it - it demonstrates
that one specific, real failure mode (single-shot closed-form inversion) is
avoidable and locates the property that avoids it. That failure mode is
still the one industrial inspection is most exposed to, since motion blur
and defocus - exactly what a naive Wiener step gets reached for - dominate
real inspection settings.

Full numbers, the per-config table, the nsr sweep figure, and the two
methodological findings about DRemR and DRR/residual-correlation are in
README.md's "The claim" - read that before deciding what (if anything) still
needs testing.

**Two things built on top of this mechanism since the above was written, both
also CPU-only and both answered, not open:** the L_pres ablation (does the
preservation loss add anything beyond `x_cons`'s structural path? yes,
established via bootstrap CI) and the Step 3 detection grid (does preserving
the residual help detection AUROC? established as detector-dependent — yes
for DIVIDE on PatchCore, no on PaDiM, in the `n_train=16` regime tested).
**`FINDINGS.md` is the authoritative writeup of both** — read it before
running the ablation or grid commands later in this document, since both
have already run once on CPU and the commands below largely re-confirm at
GPU/full scale rather than answer from a blank slate.

## Setup

```bash
git clone <this-repo-url> divide
cd divide
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` pins several things that look arbitrary and aren't — `anomalib[core]` (not bare `anomalib`), `timm<=1.0.3`, `pandas<3`, `setuptools<81` — each is commented with the exact failure it prevents. Do not loosen these without re-running the full test suite; see "Known fragile points" below for what happens if you do.

Confirm the environment before touching any data or GPU:

```bash
./scripts/smoke.sh
```

This runs all 300+ unit tests, a tiny end-to-end DRR study, a tiny end-to-end evaluation-grid run (real PaDiM fit + score, on synthetic images), a tiny PCIM training run including checkpointing, a tiny L_pres ablation, tiny DBDE validation figures, a deep-restorer registry check (fails loud if a restorer's weights are absent, runs and checks sane output if present - NAFNet/Restormer's are present on this project's dev machine now), and a Gradio Blocks construction check — entirely on CPU, in a few minutes. It must print `ALL GREEN` before you do anything else. If it doesn't, stop and fix that first; nothing downstream is trustworthy otherwise.

### Data

Neither dataset can be fetched automatically.

- **MVTec AD** — https://www.mvtec.com/company/research/datasets/mvtec-ad — accept the CC BY-NC-SA license, download the tar (~5 GB), extract it.
- **DTD** (source textures for synthetic anomalies) — https://www.robots.ox.ac.uk/~vgg/data/dtd/ (~600 MB), extract it.

Put both under one directory and point `DIVIDE_DATA_ROOT` at it:

```bash
export DIVIDE_DATA_ROOT=/path/to/parent
```

Expected layout:

```
$DIVIDE_DATA_ROOT/
  mvtec_ad/bottle/train/good/000.png          (or mvtec/, mvtec-ad/, MVTec_AD/, mvtec-anomaly-detection/ - see src/utils/paths.py)
  dtd/images/banded/banded_0002.jpg
```

MVTec's tar extracts to bare category folders (`bottle/`, `cable/`, ...) with no wrapping directory of its own — whatever you name that directory when you place it under `$DIVIDE_DATA_ROOT`, it needs to be one of the names `mvtec_root()` in `src/utils/paths.py` recognizes (I added the plain `mvtec` spelling during this pass; add another one there if you use something else, rather than renaming your actual data directory).

Verify it worked:

```bash
python -c "from src.utils.paths import require_data; print(require_data('mvtec')); print(require_data('dtd'))"
```

Both calls either print a path or raise `FileNotFoundError` with the exact URL and expected layout — there's no silent partial-success state.

## Deep restorer weights — do this by hand

NAFNet and Restormer's weights are on Google Drive and genuinely cannot be fetched with `wget`/`curl`/`gdown` reliably from a script — download them through a browser, on your own machine, then transfer them to the GPU box. DiffBIR's weights are on Hugging Face and *can* be fetched programmatically.

**NAFNet and all three Restormer checkpoints (denoising, motion deblur, defocus deblur) are already present on this project's CPU dev machine and have been exercised heavily** — the go/no-go study, the learned-deblurring test, and the Step 3 detection grid's `restormer_deblur` restorer all ran against these exact files (`FINDINGS.md`, README's "The claim"). Only DiffBIR remains genuinely unfetched and unrun anywhere in this project. If you're setting up a *new* machine, the table below still applies; if you're continuing from this project's existing `checkpoints/` directory, only DiffBIR is missing.

| Restorer | File | URL | Save as |
|---|---|---|---|
| NAFNet | `NAFNet-SIDD-width64.pth` | https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view | `checkpoints/NAFNet-SIDD-width64.pth` |
| Restormer (denoising) | `real_denoising.pth` | https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view | `checkpoints/real_denoising.pth` |
| Restormer (motion deblur) | `motion_deblurring.pth` | Google Drive **folder**: https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK — `gdown --folder` (the single-file `uc?id=` trick doesn't work on a folder) | `checkpoints/motion_deblurring.pth` |
| Restormer (defocus deblur) | `single_image_defocus_deblurring.pth` | Google Drive **folder**: https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4 — contains BOTH single-image and dual-pixel variants; grab `single_image_defocus_deblurring.pth` specifically, **not** `dual_pixel_defocus_deblurring.pth` (that one needs 6-channel stereo input, a different task) | `checkpoints/single_image_defocus_deblurring.pth` |
| DiffBIR (IRControlNet) | `v2.pth` | https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth | `checkpoints/v2.pth` |
| DiffBIR (SD v2.1 base — also required) | `v2-1_512-ema-pruned.ckpt` | https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt | `checkpoints/v2-1_512-ema-pruned.ckpt` |

DiffBIR needs *both* of its rows — the IRControlNet weights alone are not a complete model, they condition a base Stable Diffusion checkpoint.

You only need to place files under `checkpoints/`. On first use, `src/models/deep_restorers.py` clones each restorer's actual code repo into `third_party/<name>/` itself (that part *is* a plain git clone, no Google Drive involved, so it happens automatically given network access) and copies your checkpoint file into wherever that repo's own CLI expects it. NAFNet additionally needs a one-time local install inside its cloned repo — `cd third_party/nafnet && python setup.py develop --no_cuda_ext` — because it vendors its own fork of BasicSR that isn't pip-installable as-is. If you'd rather review the code before it clones anything, read `REPO_SPECS` and `build_deep_restorer()` in `src/models/deep_restorers.py`.

Confirm each one is wired up:

```bash
python -c "
from src.models.restorers import get_restorer
import numpy as np
img = np.random.rand(256, 256, 3).astype('float32')
for name in ['nafnet', 'restormer', 'diffbir']:
    try:
        out = get_restorer(name)(img)
        print(name, 'OK', out.shape)
    except RuntimeError as e:
        print(name, 'NOT READY:', str(e)[:200])
"
```

`nafnet` and `restormer` (denoising) have since run this way successfully, repeatedly, on CPU — only `diffbir` remains genuinely untested, since its weights were never fetched. Expect friction on `diffbir` specifically on first attempt; the most likely failure modes are listed under Troubleshooting.

## Run order

Run these in order. Each step names its expected wall-clock cost on a mid-range GPU (a single RTX 3090/4090-class card — scale accordingly), what files it produces, and what a correct result looks like, so a silent failure doesn't get mistaken for progress.

**1. Re-confirm the environment on this machine specifically.**

```bash
./scripts/smoke.sh
```
*Cost:* under a minute, CPU only. *Output:* console only, no files. *Correct:* `ALL GREEN`. If this fails here but passed on the laptop, it's this machine's environment (see Troubleshooting), not new code.

**2. THE GO/NO-GO EXPERIMENT - ALREADY DONE, on CPU, no GPU needed.** The
steps below describe how this was originally planned to run; it actually ran
on CPU this session (NAFNet/Restormer both run in-process on CPU, ~2-7s/image
- see `src/models/deep_restorers.py`), was carried through two follow-up
studies, and is written up with every number in README.md's "The claim" -
read that first. DiffBIR is the only piece here that genuinely still needs a
GPU. Keep reading this section only if you want to re-run or extend the
study; it's not a gate you still need to pass.

```bash
python -m src.experiments.drr_study --categories carpet bottle screw \
    --restorers identity gaussian bilateral nlm clahe msrcr wiener classical_pipeline nafnet restormer diffbir
```

*Cost:* the classical restorers are fast (CPU-bound, a few minutes total); the deep ones dominate — NAFNet/Restormer are sub-second per image, DiffBIR is a full diffusion sampler (~50 steps) and will be by far the slowest, likely tens of minutes for this whole run depending on image count. Expect roughly 1–2 GPU-hours total; if DiffBIR is unbearably slow, run it in a separate pass with fewer images first.

*Output:* `results/<tag>.csv` (per-image rows), `results/<tag>_summary.csv`, `results/<tag>_verdict.json`, and three tag-suffixed figures under `figures/` (`--tag` defaults to `drr_study`) — `drr_frontier_<tag>.png` is the one to look at first.

*Correct result looks like:* the console prints a `DEFECT RETENTION BY RESTORER` table and a `VERDICT: GO / NO-GO / MARGINAL` line with a stated reason. A run that silently drops nafnet/restormer/diffbir from the restorer list (check the printed `restorers :` line at the top of the output) without an error message means their weights weren't found — that is a **wrong, misleadingly optimistic run**, not a valid one, because the whole point of this study is measuring what *learned* restorers do; the classical baselines alone were already run on the laptop and are not the verdict. If a restorer's weights are missing, the script prints its exact `RuntimeError` to stderr and drops it from that point forward — re-check the fix and re-run rather than accepting a partial verdict.

**STOP HERE.** Read `results/<tag>_summary.csv`, the scratch-only breakdown, and `figures/drr_frontier_<tag>.png`. Send me these back before writing or running anything past this point.

- **relative DRR < 0.5** → GO, restorers substantially erase defects, the thesis holds.
- **relative DRR > 0.8** → NO-GO, restorers largely preserve defects, the premise doesn't hold here.
- **in between** → MARGINAL, look at the per-restorer and scratch-only tables before either of us decides anything.

A NO-GO is not a failure of your run — it is a legitimate, useful result, and the honest thing to do with it is report it exactly as measured and stop, not keep adjusting restorers/severities/anomaly types until the number looks like GO. If you notice yourself doing that, stop and tell me instead. See the README's Plan B note for what a NO-GO means for next steps — but that conversation happens after you send me this result, not before.

*(Everything below this line assumes a GO or MARGINAL verdict and my explicit go-ahead to continue.)*

**3. Frozen-detector harness — sanity check on real data.**

```bash
python -c "
from src.data.mvtec import load_train_normals, load_split
from src.detect.harness import DetectorHarness
normals = load_train_normals('bottle', size=256, limit=32)
test = load_split('bottle', 'test', size=256, limit=16)
h = DetectorHarness('padim', image_size=256)
h.fit(normals)
for s in test[:4]:
    r = h.score(s.image)
    print(s.defect, s.label, r.score)
h.close()
"
```

*Cost:* a couple of minutes. *Correct:* label-1 (defective) samples should generally score higher than label-0 (good) ones — not guaranteed for every single image, but PaDiM on MVTec is a well-established, easy baseline, so if scores look essentially random, something's wrong (wrong category, wrong image path, or one of the flaky-environment issues below). Repeat for `patchcore`, `reverse_distillation`, and `efficientad` — only PaDiM was reliably verified in my environment; see "What is verified" below for exactly what that means for the other three.

**4. Evaluation grid — real data, all four detectors, the restorer set from step 2.**

**A reduced-scope version of this step has already run, on CPU, real MVTec** — PaDiM + PatchCore only, 5 restorers, 4 families, severities 2–4, 3 categories, `n_train=16`/`n_test=16` per category, 360 cells (`FINDINGS.md` §3). It is gated by a PatchCore harness-reproduction check at published train/test scale (`results/patchcore_repro_check.json`, mean AUROC 0.963) confirming the harness itself is sound, and every headline `gap_closed` number has a bootstrap CI. The command below is the full-scope, GPU-target version (all 4 detectors, all restorers, all families/severities, full train/test splits) — still not run; read `FINDINGS.md` §3.2 before assuming the reduced-scope result above extends to it, since the reduced scope's own finding is explicitly scoped to `n_train=16` and explained by a mechanism (PaDiM's per-patch Gaussian being rank-deficient at that sample size) that may not apply once the training set is full-size.

```bash
python -m src.experiments.eval_grid --categories carpet bottle screw \
    --detectors padim patchcore reverse_distillation efficientad \
    --restorers identity gaussian nlm clahe wiener classical_pipeline nafnet restormer diffbir divide \
    --families defocus motion illumination noise jpeg mixed --severities 1 2 3 4 5
```

Drop `divide` from `--restorers` until step 6 below actually produces `checkpoints/pcim.pt` — it fails loudly (not silently) if the checkpoint is missing, so leaving it in early just means a `RuntimeError` gets printed and that restorer is skipped for that run.

*Cost:* embarrassingly parallel across cells but not currently parallelized in code — with 3 categories × 4 detectors × 6 families × 5 severities × 10 restorers, and each detector fit amortized once per category, expect this to be the single longest job in the pipeline (the README's original estimate for a full grid was 10–15 GPU-hours; treat that as a floor, not a ceiling, until you've timed a few cells). It is resumable — rerunning the identical command after a crash or a `Ctrl-C` skips every `(restorer, detector, category, family, severity)` cell already written to the CSV and picks up where it left off. If you need to run it in chunks, split by `--categories` or `--detectors` across separate invocations; they write to the same CSV safely as long as they don't run concurrently.

*Output:* `results/eval_grid.csv` (per-image), `results/eval_grid_summary.csv` (AUROC/`gap_closed()` per cell). *Correct:* the printed `GAP_CLOSED BY RESTORER` pivot table is non-empty (an empty one — as I hit once during development — means every cell's test subset ended up all-one-class; `_balanced_test_subset()` in `eval_grid.py` is supposed to prevent that, but re-check if it recurs) and `gap_closed` values are mostly in a sane range (roughly 0–1, though values outside that range are possible and worth reporting, not clamping away — see `metrics.core.gap_closed`'s docstring).

**5. Train PCIM.** The training script now exists (`src/experiments/train_pcim.py`) — this section used to say "write it," it no longer does. Config-driven (nothing hardcoded), resumable (kill it anytime, rerunning the same command picks up from the last checkpoint and continues the *exact same sample sequence*, not just the same weights — see "Known fragile points"), CSV-logged, verified in `--smoke` mode including an actual kill-and-resume test (`tests/test_train_pcim.py`). It has never trained on real data or a GPU.

```bash
python -m src.experiments.train_pcim --config configs/train_pcim_gpu.yaml --device cuda
```

*Cost:* unknown on GPU — time a short run first (`--smoke` for wiring, then a few hundred real steps) rather than trusting `configs/train_pcim_gpu.yaml`'s `train.steps` blindly; unlike the CPU config below, it has not been empirically calibrated against real hardware. **Loss magnitude imbalance is now handled, not just flagged**: `L_freq`'s raw magnitude ran 1-2 orders of magnitude above the other three terms in every run (FFT-magnitude L1, dominated by DC-adjacent bins), which would have meant 8 hours of optimisation for spectral fidelity and nothing measurable about `L_pres`. `loss.normalize` (default on) measures each term's raw magnitude on the first real batch and scales all four to a common baseline before the configured `w_*` weights apply - both `<term>_raw` and `<term>_scaled` are logged in `results/<run>_losses.csv` so the balance is visible, not asserted. Verified: raw magnitudes spanning ~1.6-2685 scale to ~1.0 at calibration and stay within roughly the same order of magnitude afterward. Still worth glancing at the CSV early in a real run - the calibration batch is one sample of noise, not a guarantee.

Save the trained state dict to `checkpoints/pcim.pt` (the script does this automatically via its checkpoint) — `src/models/divide_restorer.py` expects exactly that filename. SARG (`checkpoints/sarg.pt`, optional) has losses but no training script wiring it in yet; `divide_restorer.py` uses it if present and falls back to `x_full` alone if not.

**The overnight CPU run of `configs/train_pcim_cpu.yaml` referenced above completed** (128×128, 15000 steps, `results/train_pcim_cpu_eval.csv`):

> **Overnight CPU run result: relative_drr 1.201, dremr 0.001, psnr_normal 25.69 dB (n=24) at step 15000.** Per-kind: texture 1.111, scratch 1.411, blob 1.148. The loop converges and learns something real on real-sized images — this specific number is 128×128/CPU/one run, not a GPU-scale verdict, but it stopped being "pending" and became the base checkpoint the L_pres ablation below was actually built on top of.

**The L_pres ablation has since run to completion and is answered, not still open.** `src/experiments/ablate_lpres.py` trained PCIM twice from `configs/train_pcim_cpu.yaml` (identical seed/data/steps, differing only in `loss.use_lpres`, asserted in code), then both final checkpoints were re-evaluated — inference only — on the full 150-example held-out set with a 2000-resample bootstrap CI on the ON−OFF delta. **Result: established.** Scratch residual-correlation delta 95% CI [0.188, 0.296], excludes zero — L_pres adds real, statistically supported shape fidelity beyond what `x_cons` alone provides, at a cost of 1.64 dB PSNR (95% CI [1.08, 2.22] dB). Full numbers: `FINDINGS.md` §2. This was run on CPU, not the GPU config below — a GPU/`configs/train_pcim_gpu.yaml` re-run at full 256×256 resolution has not been done and could move these numbers.

```bash
python -m src.experiments.ablate_lpres --config configs/train_pcim_gpu.yaml
```

*Output:* `results/ablate_lpres_result.json` plus the four usual CSVs (`_lpres_on`/`_lpres_off` × losses/eval). *Correct:* a positive `relative_drr_delta` (L_pres ON minus OFF) is the actual evidence for the paper's central novelty claim — L_pres does something beyond what x_cons's structural preservation already provides. A delta near zero or negative is a legitimate, reportable result (it would mean the structural path is carrying all the preservation benefit and L_pres isn't adding anything measurable) — report it as measured, the same as the go/no-go verdict. (The CPU run already gave a clearly positive, bootstrap-established delta — see above; this GPU command is for confirming it holds at full resolution, not for answering the question from scratch.)

**6. Re-run step 2 and step 4 with `divide` included**, once `checkpoints/pcim.pt` exists, to get DIVIDE's own numbers into the same tables as everything else.

**7. The demo — rebuilt since this section was first written, and already run against real weights on CPU.** It is no longer a live-inference app: `src/demo/precompute.py` runs PatchCore + wiener/restormer_deblur/`divide_lpres_on` once, against real MVTec images with pasted synthetic defects (21 examples: 7 category/defect-kind/family scenes x 3 severities), and writes `demo_data/manifest.json` plus the images beside it. `src/demo/app.py` and `src/demo/static_demo.py` (the no-server HTML fallback, `demo.html`) both only read that manifest — no model runs when either is viewed. This was a deliberate redesign, not a bug fix: a live 4-panel demo re-running PCIM/Restormer/PatchCore per slider move would be slow and could crash mid-presentation on a missing weight; a precomputed lookup cannot.

```bash
python -m src.demo.precompute      # once, offline - ~7 minutes on this CPU machine, real MVTec + real weights
python -m src.demo.app             # instant, lookups only
python -m src.demo.static_demo     # writes demo.html, no server needed to view it
```

*Correct:* `precompute.py` prints one `<combo_id> done` line per example (21 total) and ends with `demo_data/manifest.json` written; `app.py` opens a page with a single dropdown (21 precomputed examples) and four panels (degraded / wiener / restormer_deblur / DIVIDE), each showing the restored image, PatchCore heatmap, score, relative DRR, and residual correlation — switching examples should be instant, since it's a dictionary lookup, not inference. One example (`bottle_scratch_defocus_sev4`, the default selection) is the deliberately-chosen striking case; read its on-screen note before assuming the panels "just look the same" - the residual-correlation numbers (wiener 0.008 vs DIVIDE 0.883) are the actual evidence, not a glance at the image alone, which is itself the point (see `FINDINGS.md` Sec. 1). All three restorer weights this demo needs were already present on this dev machine; a from-scratch setup needs the same checkpoints as step 2 above.

## What is verified vs. what has never run

Be precise about this — the 330+ passing tests cover shapes, interfaces, and logic, not model convergence or restoration quality, and it would be easy to over-trust a green `smoke.sh`.

**Actually run and verified correct, on CPU, on real and/or synthetic data:**
- Degradation simulator, synthetic anomaly generator, DBDE, all metrics (`drr`, `relative_drr`, `auroc`, `gap_closed`, ...) — this is the pre-existing, well-tested core.
- **DBDE validation, on real MVTec (carpet/bottle/screw, 15 images/category, 256×256) — see the "CPU-preliminary results" section below for the actual numbers and `figures/dbde_*.png`.** This is genuinely finished, not preliminary in the same sense as the deep-restorer/DIVIDE numbers — DBDE needs no GPU and nothing here changes when it moves to a GPU machine.
- **The full classical DRR study, on real MVTec, all 3 categories × 6 families × 5 severities × 3 anomaly kinds × 8 classical restorers** — see below for the result and its status as a baseline, not the verdict.
- PCIM: every code path, including the structural linearity claim for `x_cons` (`test_hqs_without_prox_is_linear`), gradients, parameter count, shapes at n_iters ∈ {4,5,6}.
- Losses: all four, including `L_pres` against the real `degrade_pair()`/`paste_anomaly()` pipeline, not just hand-built tensors.
- SARG: RPCA primitives, mask behavior, blending, gradients.
- The classical restorer registry, the deep-restorer and DIVIDE fail-loud paths (real `RuntimeError`s, real messages, confirmed to name the actual missing file).
- The frozen-detector harness: **PaDiM and PatchCore, both on real MVTec, CPU** — fit + score, both classes, correct anomaly-map shape, repeated many times including inside the full test suite and across the full 360-cell evaluation grid (`FINDINGS.md` §3), plus a dedicated PatchCore published-config reproduction check (`results/patchcore_repro_check.json`) confirming the harness matches its published AUROC at full train/test scale. ReverseDistillation and EfficientAd remain fit+scored at least once each in an isolated process, not through the full grid.
- The evaluation grid runner end-to-end on **real MVTec** with PaDiM and PatchCore (`FINDINGS.md` §3, reduced `n_train=16`/`n_test=16` scope), not just synthetic data — including a genuine resume check (rerunning adds zero duplicate rows) and a reconciliation step for the image/pixel-CSV pair (`_reconcile_partial_cells`, `src/experiments/eval_grid.py`) that handles a crash between writing a cell's image rows and its pixel-level row.
- **The demo, end to end, against real weights** — `precompute.py` has run to completion on real MVTec with real wiener/restormer_deblur/DIVIDE checkpoints (21 examples, `demo_data/manifest.json`), `app.py`'s Blocks construction is tested against that real manifest (not just a fake one), and `static_demo.py`'s `demo.html` has been generated and its embedded-image count checked. Not verified: what it looks like in an actual browser window (no running-server / visual check was done, only that construction and HTML generation don't raise and produce structurally correct output).
- **The PCIM training loop's wiring and resumability** (`src/experiments/train_pcim.py`): `--smoke` mode runs end to end, and a real kill-and-resume test confirms a killed-and-restarted run produces the *identical* loss trajectory a same-seed uninterrupted run would (`tests/test_train_pcim.py::test_resumed_run_continues_the_same_sample_sequence_as_uninterrupted`) — this is the property the ablation harness's fairness guarantee depends on, so it's tested directly rather than assumed.
- **Loss magnitude normalisation**: `calibrate_scale_factors()` is tested to actually bring wildly different raw magnitudes to the same order of magnitude, `loss.normalize: false` is tested to be a true no-op, and scale factors are tested to persist across a checkpoint resume rather than being recomputed.
- **Per-anomaly-kind relative DRR in the eval loop** (`relative_drr_texture`/`_scratch`/`_blob` in `results/<run>_eval.csv`) — tested to be present and finite-or-nan for every kind.
- **The L_pres ablation harness's fairness guarantee** (`src/experiments/ablate_lpres.py`): `_assert_only_use_lpres_differs` is tested to actually raise when the two variant configs drift in anything besides `loss.use_lpres`, and to pass when they don't.

**Constructed correctly to the best of my research but never executed, because I have no GPU:**
- NAFNet, Restormer, DiffBIR loaders — the subprocess/CLI commands are transcribed from each repo's README as of this writing; repos change, and I could not run a single one to confirm the exact flags still match. (Restormer's denoising *and* deblurring checkpoints, plus NAFNet, did end up getting exercised heavily on CPU this session — see below — so this caveat now applies mainly to DiffBIR and to whether the CLI commands still match upstream, not to whether the loaders work at all.)
- ReverseDistillation, EfficientAd through the frozen-detector harness — each **did** fit and score correctly at least once during development, in an isolated process, but this exact sandbox (Python 3.14 — PyTorch itself warns this is unsupported) produced non-reproducible failures across repeated identical runs at the time this was written. I could not fully root-cause this; see "Known fragile points" below. **PatchCore no longer belongs in this list** — it has since run reliably and reproducibly through the full 360-cell evaluation grid plus a dedicated full-scale reproduction check, in this same sandbox, with no recurrence of the non-determinism (`FINDINGS.md` §3.1). Treat ReverseDistillation/EfficientAd as "structurally correct, not yet trustworthy" until similarly exercised.
- ~~PCIM training and the L_pres ablation, at any meaningful scale.~~ **No longer true — both ran to completion on CPU and are answered, not open questions.** PCIM's overnight CPU run converged (see run-order section 5); the L_pres ablation ran two full training runs and was re-evaluated with bootstrap CIs on the full 150-example held-out set — established (`FINDINGS.md` §2). What remains genuinely untested is the *GPU-scale, 256×256* version of both — the CPU numbers are real results, not preliminary noise, but they are not GPU-resolution numbers.
- DIVIDE as a restorer, the demo against real weights — DIVIDE-as-a-restorer has since run, on CPU, against its own real trained checkpoints (`src/models/divide_restorer.py::build_divide_restorer_from_run`, exercised throughout `FINDINGS.md` §3 and §2's bootstrap comparison) — this bullet no longer applies to it. The demo against real weights is addressed in step 7 below.

## CPU-preliminary results already gathered

These ran on this laptop, on real MVTec data, and are genuine results — but read the caveats under each before citing them as "the" result for anything.

**DBDE validation (`figures/dbde_defect_blindness.png`, `figures/dbde_parameter_accuracy.png`, `figures/dbde_reference_vs_blind.png`; carpet/bottle/screw, 15 images/category, 256×256):**
- *Defect blindness is essentially exact*, not just "roughly flat" — estimation drift from a pasted defect is precisely 0 across the entire 0.2%–10% area-fraction range for both defocus radius and motion length, and negligible for noise sigma (max drift 0.00017, against a sigma range of 0–0.06). This is a final result, not preliminary — nothing about running on a GPU changes a classical estimator's defect-blindness.
- *Reference PSD estimation is dramatically more accurate than blind*, especially for subtle blur: reference-based hits ~0 MAE at every severity tested; blind is close except at the smallest defocus radius (1px true), where it's off by ~0.17px. Also final — this doesn't depend on any learned component.
- *Noise sigma is systematically underestimated at higher severities* — the estimated value tracks roughly half the true sigma once severity increases (see `dbde_parameter_accuracy.png`, rightmost panel). This is a real calibration gap in `estimate_noise_sigma`'s flat-patch PCA approach, not a fluke of one run; worth either fixing (recalibrating the eigenvalue-quantile logic) or at minimum accounting for in anything downstream that consumes DBDE's noise estimate directly (PCIM's VST does — see "Known fragile points").

**Classical DRR study, real MVTec, full grid** (`results/drr_study_real.csv`, `results/drr_study_real_summary.csv`, `figures/drr_frontier_drr_study_real.png` etc. — tag-suffixed, does not collide with the go/no-go study's own figures):

Ran to completion: 14,400 rows (3 categories × 20 images × 6 families × 5 severities × 8 classical restorers), script-reported verdict **NO-GO** (mean relative DRR 1.002 > 0.8, ratio-of-means — see README's "The claim" for why that convention and not a per-example-averaged one) — but the mean hides a real, non-obvious split that the per-restorer table shows clearly:

| restorer | relative DRR (ratio-of-means) | reading |
|---|---|---|
| msrcr | 1.60 | amplifies the residual (contrast-enhancement artefact, not real preservation) |
| clahe | 1.57 | same |
| identity | 1.00 | baseline, by construction |
| bilateral | 0.98 | barely touches the defect |
| nlm | 0.96 | barely touches the defect |
| gaussian | 0.90 | mild erosion |
| wiener | 0.51 | erases roughly half the defect residual |
| classical_pipeline | 0.49 | same |

The plain denoisers (bilateral/nlm/gaussian) sit close to 1.0 across every severity (`figures/drr_vs_severity_drr_study_real.png`) — spatial smoothing alone doesn't do much to a scratch or blob at these degradation levels. The deconvolution-based methods (wiener, and `classical_pipeline`, which chains illumination correction, denoising, and Wiener deblurring) sit consistently around 0.5 — deconvolution's ringing/sharpening measurably suppresses fine defect structure, most visibly for scratches specifically (`figures/drr_by_kind_drr_study_real.png`). clahe/msrcr's numbers above 1.0 are not "better than identity" in any meaningful sense - their contrast stretching inflates the raw pixel-difference metric on both normal and defect regions alike (see their `dremr_mean` in `results/drr_study_real_summary.csv`, both strongly negative - they move *further* from the clean image than the degraded input already was). This is the same split that later analysis (see README's "The claim") traced to a mechanism - deconvolution vs. denoising - and then to unrolled iteration vs. one-shot inversion specifically, using exactly this wiener/classical_pipeline result as the erosion anchor.

At the time this was written, this was **not the go/no-go verdict** — classical restorers only, with the note that classical restorers alone can't settle the question the project is actually about. **That measurement has since happened** (`results/drr_study_gonogo*`, see "What this project claims" above): a real learned denoiser (Restormer) preserves defects the same way the classical denoisers here do, while classical deconvolution erodes them the same way it does here — confirming the split predicted by this table is about mechanism (deconvolve vs. denoise), not learned vs. classical. NAFNet's result in that run doesn't extend this table's pattern one way or the other - it was out-of-distribution on synthetic degradations (see the caveat in "What this project claims").

## Known fragile points

Two bugs were already found and fixed before I started (see the README's "Two bugs worth knowing about" section) — both were the same shape of failure: **a step that looks like an unrelated, reasonable choice quietly breaks a physical or statistical invariant the rest of the system depends on, with no symptom other than "the numbers don't look right" or "training won't converge."** Watch for more of these, especially anywhere you touch:

- **The counterfactual pair (`degrade_pair()`).** `L_pres` needs the with-anomaly and without-anomaly images to receive *identical* degradation parameters and *identical* noise realizations. If you refactor `apply_degradation()` or how randomness flows through it, re-run `test_degrade_pair_shares_parameters_and_noise` and `test_preservation_loss_with_real_counterfactual_pair` (in `tests/test_losses.py`) — a break here has no visible symptom beyond a training run that mysteriously plateaus, which is exactly how the original noise-ordering bug was found.
- **DBDE's illumination fit absorbing a defect into the estimated field.** If you ever reintroduce a pre-blur before the robust polynomial fit, re-run `test_illumination_degree_cannot_fit_a_scratch` — this is the second already-fixed bug, and the fix (fit the raw log image with Huber weights, only median-prefilter for sensor noise) is easy to accidentally regress by "simplifying" the estimator.
- **anomalib vs. a current Python/numpy/pandas/rich/matplotlib stack.** Four separate runtime shims in `src/detect/_compat.py`, plus four requirements.txt pins, were needed just to get PatchCore/PaDiM/ReverseDistillation/EfficientAd to import and fit at all (numpy 2.0 removed `np.sctypes`, which `imgaug` — pulled in by `anomalib[core]` — reads at import time; `anomalib.models.__init__` eagerly imports an unrelated video model whose legacy CLIP loader needs `pkg_resources`, dropped by `setuptools>=81`; a Matplotlib API rename broke anomalib's prediction-time visualizer; a private `rich.Console` attribute rename broke PatchCore's coreset progress bar). None of this is DIVIDE logic — it's dependency archaeology across a two-year gap — but if you upgrade any of numpy/pandas/rich/matplotlib/anomalib, re-run `./scripts/smoke.sh` before trusting the result, and read the docstrings in `_compat.py` for what each shim is actually patching.
- **The intermittent `TypeError: only 0-dimensional arrays can be converted to Python scalars`.** I hit this from `PIL.ImageOps` (via anomalib's synthetic-anomaly augmentation pipeline, triggered by `test_split_mode="synthetic"`) non-deterministically — identical code, identical input, sometimes passed, sometimes didn't, across back-to-back runs of the full test suite. I worked around it by not using `test_split_mode="synthetic"` at all (`DetectorHarness.fit()` now carves its validation split out of the normal images directly instead — see the comment in `src/detect/harness.py`), which made 3 consecutive full-suite runs pass identically. If this resurfaces on your machine in a code path that still uses `synthetic` mode, or on a normal (non-3.14) Python, it's worth actually root-causing rather than routing around again — I'd treat this as the single most likely source of confusing results anywhere the detector harness is involved.
- **ReverseDistillation needs `image_size` to be a multiple of 32** (its encoder/decoder need matching spatial sizes at every downsampling scale) — 48 fails with a tensor-size mismatch, 64 works. PaDiM and PatchCore don't have this constraint. `eval_grid.py --smoke` and the demo both default to 64 for this reason.
- **EfficientAd downloads the Imagenette dataset (~768 MB) on first fit**, into `datasets/imagenette/` — it needs this for its natural-image regularization term. This is the one detector that needs more than just the category's own normal images, and it needs network access the first time even though you're "just fitting on normals."
- **`reference_psd()` only means what it claims if the reference images and test images are the same part.** Caught this one myself while building `dbde_validation.py`: an early version pooled all three categories into one shuffled list before splitting into reference/test, so a carpet reference PSD occasionally got compared against a screw test image. The result looked like "reference-based blur estimation is dramatically WORSE than blind" (a −2133% "improvement"), which contradicted both the module's docstring and basic intuition — exactly the kind of result that's tempting to write down as a surprising finding instead of a bug. Fixed by keeping reference and test images within one category (`reference_vs_blind_data()`'s docstring states the requirement now); the real result (reference dramatically better, not worse) is in the "CPU-preliminary results" section above. If you extend this comparison anywhere else, keep the same-category invariant in mind.
- **DBDE's noise-sigma estimate is systematically low at higher severities** (see "CPU-preliminary results" above) — roughly half the true value once severity increases. PCIM's generalised Anscombe VST (`src/models/pcim.py`) consumes this estimate directly and has never been trained, so it has never had a chance to compensate for this bias through learning. If PCIM's denoising looks conspicuously weak at high severities once you do train it, check whether this is the reason before assuming it's a PCIM architecture problem.
- **PCIM's loss terms had a real magnitude imbalance — now fixed, not just noted.** `L_freq` (FFT-magnitude L1, unnormalized) came out one to two orders of magnitude larger than `L_rec`/`L_deg`/`L_pres` in every CPU run, dominated by DC-adjacent bins; left alone, hours of optimisation would have gone almost entirely to spectral fidelity, telling you nothing about `L_pres`. `train_pcim.py` now measures each term's raw magnitude on the first real batch and computes per-term scale factors (`calibrate_scale_factors()`) before the configured `w_*` weights apply — behind `loss.normalize` (default true), stored in the checkpoint so a resumed run doesn't recalibrate from a different batch. Both `<term>_raw` and `<term>_scaled` are logged per step. This is a first-batch calibration, not a guarantee for every subsequent batch — if one term still visibly dominates `*_scaled` deep into a real run, that's a genuine finding (that term may be harder to reduce, not just larger), not evidence the fix failed.
- **FIXED, was a real recurring problem: `drr_study.py`'s figure filenames used to not be tag-prefixed.** `drr_frontier.png`/`drr_vs_severity.png`/`drr_by_kind.png` used to be written under those exact names regardless of `--tag`, so running `./scripts/smoke.sh` (or any other `drr_study.py` invocation, smoke or real) after generating real-data figures silently overwrote them in the working tree - hit repeatedly across this project's sessions, including twice finishing the original version of this handoff. `make_figures()` now takes a `tag` argument and suffixes all three filenames with it (`drr_frontier_<tag>.png` etc); `main()` passes `--tag` through automatically. The old unsuffixed files are gone from the repo, replaced by tag-suffixed ones per study (`drr_study_gonogo`, `drr_study_deblur`, `drr_study_deblur_corr`, `wiener_bridge_ablation`, `drr_study_real`). If you call `make_figures()` directly without a tag it still writes the old unsuffixed names - pass one.
- **The PCIM training script's resumability is only as good as its RNG-state save/restore, and that's numpy-Generator-specific.** `save_checkpoint()`/`load_checkpoint()` in `train_pcim.py` persist `rng.bit_generator.state` for the one `numpy.random.Generator` that drives all data sampling (`paste_anomaly`, family/severity choice, `degrade_pair`'s seed). If you ever introduce a second independent source of randomness into the training loop (e.g. a stochastic model component, or `torch`-side augmentation), it won't be captured by this checkpoint, and a resumed run will silently diverge from what an uninterrupted run would have done — re-run `test_resumed_run_continues_the_same_sample_sequence_as_uninterrupted` after any change near the sampling path, the same way you'd re-run the counterfactual-pair tests after touching `degrade_pair()`.

## Troubleshooting

**"ModuleNotFoundError: lightning" (or kornia, freia, open-clip-torch, ...) the moment you import anything from `anomalib.models`.** You installed `anomalib==1.1.0` instead of `anomalib[core]==1.1.0`. Re-run `pip install -r requirements.txt` as-is; don't hand-edit the extras away.

**`AttributeError: 'Console' object has no attribute '_live'`, deep inside PatchCore's coreset sampling.** A `rich` version newer than anomalib expects renamed a private attribute this shimmed in `src/detect/_compat.py::apply()`. If this fires despite the shim being applied, check that `apply()` actually ran before any `anomalib` import in your code path — it has to run first, every time, in every process; it's not persistent across process boundaries.

**A deep restorer (NAFNet/Restormer/DiffBIR) or DIVIDE raises `RuntimeError` naming a missing file even though you're sure you downloaded it.** Check the *exact* path in the error message against where you actually saved it — `checkpoints/<filename>`, flat, no subdirectories, filename must match exactly (case-sensitive on Linux). For NAFNet/Restormer specifically, also confirm the repo actually got cloned into `third_party/<name>/` (check for a `.git` directory there) and, for NAFNet, that you ran the one-time `python setup.py develop --no_cuda_ext` inside it.

**`ValueError: The truth value of a Series is ambiguous`, from inside `anomalib/data/image/folder.py`.** You have `pandas>=3`. Downgrade to the `pandas<3` pin in requirements.txt — this is a real, reproducible incompatibility I found and fixed, not a guess.

**The frozen-detector harness (anything but PaDiM) fails, passes on retry, or fails differently on retry with no code change.** Read "Known fragile points" above first — this may be the same non-deterministic behavior I saw, particularly likely if you're also on an unusually new Python version. Try three things in order: (1) just re-run it a few times and see if it's genuinely nondeterministic here too; (2) confirm you're on a standard, well-supported Python (3.10–3.12) rather than something bleeding-edge; (3) if it's still flaky, bisect which specific anomalib code path is nondeterministic (start with whatever's using randomness — augmentation, coreset sampling — since those are the most likely culprits) rather than assuming it's DIVIDE-side.

**`AssertionError: ablation configs differ in more than loss.use_lpres: [...]`, from `ablate_lpres.py`.** This is working as designed, not a bug — it means something (you, or a config edit) made the two ablation variants differ in more than the one term they're supposed to differ in, which would make the resulting `relative_drr_delta` meaningless. Fix whichever of `configs/train_pcim_*.yaml`'s settings drifted (the error names the exact keys) rather than suppressing the assertion.

**`train_pcim.py` or `ablate_lpres.py` seems to restart from step 0 instead of resuming.** Check you're passing the same `--run-name` (or letting it default to the config filename's stem) as the original run, and that `checkpoints/<run-name>.pt` actually exists and is non-empty — `save_checkpoint()` writes to a `.tmp` file and renames atomically, so a killed process should never leave a corrupt checkpoint, but a checkpoint under a *different* run-name than the one you're resuming with will silently look like "no checkpoint found" rather than an error, since a missing checkpoint is the normal state for a fresh run.

**A DBDE validation or classical-restorer script is far slower than expected on real data.** This is expected, not a bug — `estimate()` runs an FFT-based PSD, a signed cepstrum, and a Huber-IRLS illumination fit per image, none of which are free at 256×256. Measured on this laptop: the full real-MVTec classical DRR study (3 categories × 20 images × 6 families × 5 severities × 8 restorers ≈ 14,400 cells) ran at roughly 2.6 cells/second: budget accordingly rather than assuming something's hung.
