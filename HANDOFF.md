# Handoff: DIVIDE, laptop → GPU machine

You have a GPU and (presumably) no context on this project beyond this document. Read it before running anything. It tells you exactly what to run, in what order, what a correct result looks like, and where the code is likely to be wrong because I wrote it without being able to execute it.

## What this project claims

Industrial visual inspection pipelines run capture → restore → detect: a degraded photo gets cleaned up by a restoration model before a defect detector looks at it. This project argues that step is self-defeating. A restoration model — classical or deep — is fundamentally a denoiser, and a surface defect (a scratch, a stain, a dent) is statistically a sparse, spatially localized anomaly. To a model whose entire training objective is "make this image look like a clean, defect-free instance of its class," a defect *is* noise, and it gets erased along with the sensor noise, the blur, and the bad lighting. The detector then runs on an image that's been quietly cleaned of the exact evidence it needed.

DIVIDE's fix is to stop trying to restore the image and instead invert the specific physical degradation that hit it — estimate the blur kernel, the illumination field, the noise level (via DBDE, already built and CPU-tested), and apply their mathematical inverses (via PCIM). The inverse of a convolution, a multiplicative illumination field, and additive noise is a fixed, content-agnostic operation: dividing by an illumination field doesn't know or care whether the pixel underneath is a defect or a shadow. That's the whole bet — preservation becomes a structural property of the inversion, not something you hope a learned model picked up. `tests/test_pcim.py::test_hqs_without_prox_is_linear` verifies this structurally: with PCIM's learned component turned off, its core operation is provably an affine map of the input, which is what makes "it can't learn to erase this defect" a claim about the math rather than a hope.

Whether the premise is even true — whether real restoration models actually destroy defect signal on real degraded industrial images — is exactly what you're about to go measure. That's the go/no-go stop below, and it is the most important thing in this document.

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

This runs all ~225 unit tests, a tiny end-to-end DRR study, a tiny end-to-end evaluation-grid run (real PaDiM fit + score, on synthetic images), a deep-restorer fail-loud check, and a Gradio Blocks construction check — entirely on CPU, entirely on synthetic data, in well under a minute. It must print `ALL GREEN` before you do anything else. If it doesn't, stop and fix that first; nothing downstream is trustworthy otherwise.

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

| Restorer | File | URL | Save as |
|---|---|---|---|
| NAFNet | `NAFNet-SIDD-width64.pth` | https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view | `checkpoints/NAFNet-SIDD-width64.pth` |
| Restormer | `real_denoising.pth` | https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view | `checkpoints/real_denoising.pth` |
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

This has never been run — I have no GPU and no weights. Expect friction on first attempt; the most likely failure modes are listed under Troubleshooting.

## Run order

Run these in order. Each step names its expected wall-clock cost on a mid-range GPU (a single RTX 3090/4090-class card — scale accordingly), what files it produces, and what a correct result looks like, so a silent failure doesn't get mistaken for progress.

**1. Re-confirm the environment on this machine specifically.**

```bash
./scripts/smoke.sh
```
*Cost:* under a minute, CPU only. *Output:* console only, no files. *Correct:* `ALL GREEN`. If this fails here but passed on the laptop, it's this machine's environment (see Troubleshooting), not new code.

**2. THE GO/NO-GO EXPERIMENT.** Do this before anything else below — it is the entire reason to build DIVIDE or not.

```bash
python -m src.experiments.drr_study --categories carpet bottle screw \
    --restorers identity gaussian bilateral nlm clahe msrcr wiener classical_pipeline nafnet restormer diffbir
```

*Cost:* the classical restorers are fast (CPU-bound, a few minutes total); the deep ones dominate — NAFNet/Restormer are sub-second per image, DiffBIR is a full diffusion sampler (~50 steps) and will be by far the slowest, likely tens of minutes for this whole run depending on image count. Expect roughly 1–2 GPU-hours total; if DiffBIR is unbearably slow, run it in a separate pass with fewer images first.

*Output:* `results/drr_study.csv` (per-image rows), `results/drr_study_summary.csv`, `results/drr_study_verdict.json`, and three figures under `figures/` — `drr_frontier.png` is the one to look at first.

*Correct result looks like:* the console prints a `DEFECT RETENTION BY RESTORER` table and a `VERDICT: GO / NO-GO / MARGINAL` line with a stated reason. A run that silently drops nafnet/restormer/diffbir from the restorer list (check the printed `restorers :` line at the top of the output) without an error message means their weights weren't found — that is a **wrong, misleadingly optimistic run**, not a valid one, because the whole point of this study is measuring what *learned* restorers do; the classical baselines alone were already run on the laptop and are not the verdict. If a restorer's weights are missing, the script prints its exact `RuntimeError` to stderr and drops it from that point forward — re-check the fix and re-run rather than accepting a partial verdict.

**STOP HERE.** Read `results/drr_study_summary.csv`, the scratch-only breakdown, and `figures/drr_frontier.png`. Send me these back before writing or running anything past this point.

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

```bash
python -m src.experiments.eval_grid --categories carpet bottle screw \
    --detectors padim patchcore reverse_distillation efficientad \
    --restorers identity gaussian nlm clahe wiener classical_pipeline nafnet restormer diffbir divide \
    --families defocus motion illumination noise jpeg mixed --severities 1 2 3 4 5
```

Drop `divide` from `--restorers` until step 6 below actually produces `checkpoints/pcim.pt` — it fails loudly (not silently) if the checkpoint is missing, so leaving it in early just means a `RuntimeError` gets printed and that restorer is skipped for that run.

*Cost:* embarrassingly parallel across cells but not currently parallelized in code — with 3 categories × 4 detectors × 6 families × 5 severities × 10 restorers, and each detector fit amortized once per category, expect this to be the single longest job in the pipeline (the README's original estimate for a full grid was 10–15 GPU-hours; treat that as a floor, not a ceiling, until you've timed a few cells). It is resumable — rerunning the identical command after a crash or a `Ctrl-C` skips every `(restorer, detector, category, family, severity)` cell already written to the CSV and picks up where it left off. If you need to run it in chunks, split by `--categories` or `--detectors` across separate invocations; they write to the same CSV safely as long as they don't run concurrently.

*Output:* `results/eval_grid.csv` (per-image), `results/eval_grid_summary.csv` (AUROC/`gap_closed()` per cell). *Correct:* the printed `GAP_CLOSED BY RESTORER` pivot table is non-empty (an empty one — as I hit once during development — means every cell's test subset ended up all-one-class; `_balanced_test_subset()` in `eval_grid.py` is supposed to prevent that, but re-check if it recurs) and `gap_closed` values are mostly in a sane range (roughly 0–1, though values outside that range are possible and worth reporting, not clamping away — see `metrics.core.gap_closed`'s docstring).

**5. Write the PCIM training script — this does not exist yet.**

`src/models/pcim.py`, `src/models/losses.py`, `src/models/sarg.py`, and `src/degrade/simulator.py`'s `degrade_pair()` are all built and CPU-tested, but nothing wires them into an actual training loop. That's the next piece of glue, and it's yours to write — roughly: for each training step, sample a clean image and a synthetic anomaly (`src.degrade.anomaly.paste_anomaly`), get a counterfactual pair via `degrade_pair()`, run DBDE (`src.dbde.estimator.estimate`) on the degraded-with-anomaly image to get the illumination field / kernel / noise sigma PCIM needs, run PCIM forward to get `(x_full, x_cons)` for both members of the pair, compute `divide_loss()` from `src/models/losses.py`, backprop, step. Save the trained state dict to `checkpoints/pcim.pt` (`torch.save(model.state_dict(), ...)`) — `src/models/divide_restorer.py` expects exactly that filename and loads it with `model.load_state_dict()`. SARG (`checkpoints/sarg.pt`, optional) follows the same pattern if you train it too; `divide_restorer.py` uses it if present and falls back to `x_full` alone if not.

*Cost:* README's original estimate was 3–4 GPU-hours for PCIM training; that estimate predates this script existing, so treat it as a rough guide, not a verified number.

**6. Re-run step 2 and step 4 with `divide` included**, once `checkpoints/pcim.pt` exists, to get DIVIDE's own numbers into the same tables as everything else.

**7. The Gradio demo, for presenting live.**

```bash
python -m src.demo.app
```

*Cost:* seconds to start (one PaDiM fit on 16 synthetic images) — swap `CATEGORY`/`DETECTOR_NAME` at the top of `src/demo/app.py` for a real category once weights exist. *Correct:* a local URL prints and opens a page with two sliders (family, severity) and four panels; moving a slider should update all four panels in well under a second, since the detector is fit once at startup and never again (that's the whole point of "cache the memory bank"). Until `checkpoints/` has Restormer and DIVIDE weights, expect panels 3 and 4 to show their fail-loud message instead of an image — that's the correct, honest state, not a bug; the demo has never been run against real weights.

## What is verified vs. what has never run

Be precise about this — the 225 passing tests cover shapes, interfaces, and logic, not model convergence or restoration quality, and it would be easy to over-trust a green `smoke.sh`.

**Actually run and verified correct, on CPU, on real and/or synthetic data:**
- Degradation simulator, synthetic anomaly generator, DBDE, all metrics (`drr`, `relative_drr`, `auroc`, `gap_closed`, ...) — this is the pre-existing, well-tested core.
- The real (non-smoke) DRR study end-to-end on actual MVTec `bottle` images with real DTD textures — I ran a small slice of it myself on this laptop (results discarded, it was classical-restorers-only and not the real verdict).
- PCIM: every code path, including the structural linearity claim for `x_cons` (`test_hqs_without_prox_is_linear`), gradients, parameter count, shapes at n_iters ∈ {4,5,6}.
- Losses: all four, including `L_pres` against the real `degrade_pair()`/`paste_anomaly()` pipeline, not just hand-built tensors.
- SARG: RPCA primitives, mask behavior, blending, gradients.
- The classical restorer registry, the deep-restorer and DIVIDE fail-loud paths (real `RuntimeError`s, real messages, confirmed to name the actual missing file).
- The frozen-detector harness: **PaDiM specifically** — fit + score, both classes, correct anomaly-map shape, repeated many times including inside the full test suite, reliably.
- The evaluation grid runner end-to-end on synthetic data with PaDiM, including a genuine resume check (rerunning adds zero duplicate rows).
- The Gradio demo's panel logic and Blocks construction (not an actual running server — I didn't verify the browser-facing UI renders correctly, only that building it doesn't raise).

**Constructed correctly to the best of my research but never executed, because I have no GPU:**
- NAFNet, Restormer, DiffBIR loaders — the subprocess/CLI commands are transcribed from each repo's README as of this writing; repos change, and I could not run a single one to confirm the exact flags still match.
- PatchCore, ReverseDistillation, EfficientAd through the frozen-detector harness — each one **did** fit and score correctly at least once during development, in an isolated process, but this exact sandbox (Python 3.14 — PyTorch itself warns this is unsupported) produced non-reproducible failures across repeated identical runs. I could not fully root-cause this; see "Known fragile points" below. Treat these three as "structurally correct, not yet trustworthy" until you've run `tests/test_harness.py`-style checks for each of them a few times on your machine and they're consistently green.
- Everything past step 4 in the run order above (PCIM training, DIVIDE as a restorer, the demo against real weights) — untested by construction, since none of the artifacts they need exist yet.

## Known fragile points

Two bugs were already found and fixed before I started (see the README's "Two bugs worth knowing about" section) — both were the same shape of failure: **a step that looks like an unrelated, reasonable choice quietly breaks a physical or statistical invariant the rest of the system depends on, with no symptom other than "the numbers don't look right" or "training won't converge."** Watch for more of these, especially anywhere you touch:

- **The counterfactual pair (`degrade_pair()`).** `L_pres` needs the with-anomaly and without-anomaly images to receive *identical* degradation parameters and *identical* noise realizations. If you refactor `apply_degradation()` or how randomness flows through it, re-run `test_degrade_pair_shares_parameters_and_noise` and `test_preservation_loss_with_real_counterfactual_pair` (in `tests/test_losses.py`) — a break here has no visible symptom beyond a training run that mysteriously plateaus, which is exactly how the original noise-ordering bug was found.
- **DBDE's illumination fit absorbing a defect into the estimated field.** If you ever reintroduce a pre-blur before the robust polynomial fit, re-run `test_illumination_degree_cannot_fit_a_scratch` — this is the second already-fixed bug, and the fix (fit the raw log image with Huber weights, only median-prefilter for sensor noise) is easy to accidentally regress by "simplifying" the estimator.
- **anomalib vs. a current Python/numpy/pandas/rich/matplotlib stack.** Four separate runtime shims in `src/detect/_compat.py`, plus four requirements.txt pins, were needed just to get PatchCore/PaDiM/ReverseDistillation/EfficientAd to import and fit at all (numpy 2.0 removed `np.sctypes`, which `imgaug` — pulled in by `anomalib[core]` — reads at import time; `anomalib.models.__init__` eagerly imports an unrelated video model whose legacy CLIP loader needs `pkg_resources`, dropped by `setuptools>=81`; a Matplotlib API rename broke anomalib's prediction-time visualizer; a private `rich.Console` attribute rename broke PatchCore's coreset progress bar). None of this is DIVIDE logic — it's dependency archaeology across a two-year gap — but if you upgrade any of numpy/pandas/rich/matplotlib/anomalib, re-run `./scripts/smoke.sh` before trusting the result, and read the docstrings in `_compat.py` for what each shim is actually patching.
- **The intermittent `TypeError: only 0-dimensional arrays can be converted to Python scalars`.** I hit this from `PIL.ImageOps` (via anomalib's synthetic-anomaly augmentation pipeline, triggered by `test_split_mode="synthetic"`) non-deterministically — identical code, identical input, sometimes passed, sometimes didn't, across back-to-back runs of the full test suite. I worked around it by not using `test_split_mode="synthetic"` at all (`DetectorHarness.fit()` now carves its validation split out of the normal images directly instead — see the comment in `src/detect/harness.py`), which made 3 consecutive full-suite runs pass identically. If this resurfaces on your machine in a code path that still uses `synthetic` mode, or on a normal (non-3.14) Python, it's worth actually root-causing rather than routing around again — I'd treat this as the single most likely source of confusing results anywhere the detector harness is involved.
- **ReverseDistillation needs `image_size` to be a multiple of 32** (its encoder/decoder need matching spatial sizes at every downsampling scale) — 48 fails with a tensor-size mismatch, 64 works. PaDiM and PatchCore don't have this constraint. `eval_grid.py --smoke` and the demo both default to 64 for this reason.
- **EfficientAd downloads the Imagenette dataset (~768 MB) on first fit**, into `datasets/imagenette/` — it needs this for its natural-image regularization term. This is the one detector that needs more than just the category's own normal images, and it needs network access the first time even though you're "just fitting on normals."

## Troubleshooting

**"ModuleNotFoundError: lightning" (or kornia, freia, open-clip-torch, ...) the moment you import anything from `anomalib.models`.** You installed `anomalib==1.1.0` instead of `anomalib[core]==1.1.0`. Re-run `pip install -r requirements.txt` as-is; don't hand-edit the extras away.

**`AttributeError: 'Console' object has no attribute '_live'`, deep inside PatchCore's coreset sampling.** A `rich` version newer than anomalib expects renamed a private attribute this shimmed in `src/detect/_compat.py::apply()`. If this fires despite the shim being applied, check that `apply()` actually ran before any `anomalib` import in your code path — it has to run first, every time, in every process; it's not persistent across process boundaries.

**A deep restorer (NAFNet/Restormer/DiffBIR) or DIVIDE raises `RuntimeError` naming a missing file even though you're sure you downloaded it.** Check the *exact* path in the error message against where you actually saved it — `checkpoints/<filename>`, flat, no subdirectories, filename must match exactly (case-sensitive on Linux). For NAFNet/Restormer specifically, also confirm the repo actually got cloned into `third_party/<name>/` (check for a `.git` directory there) and, for NAFNet, that you ran the one-time `python setup.py develop --no_cuda_ext` inside it.

**`ValueError: The truth value of a Series is ambiguous`, from inside `anomalib/data/image/folder.py`.** You have `pandas>=3`. Downgrade to the `pandas<3` pin in requirements.txt — this is a real, reproducible incompatibility I found and fixed, not a guess.

**The frozen-detector harness (anything but PaDiM) fails, passes on retry, or fails differently on retry with no code change.** Read "Known fragile points" above first — this may be the same non-deterministic behavior I saw, particularly likely if you're also on an unusually new Python version. Try three things in order: (1) just re-run it a few times and see if it's genuinely nondeterministic here too; (2) confirm you're on a standard, well-supported Python (3.10–3.12) rather than something bleeding-edge; (3) if it's still flaky, bisect which specific anomalib code path is nondeterministic (start with whatever's using randomness — augmentation, coreset sampling — since those are the most likely culprits) rather than assuming it's DIVIDE-side.
