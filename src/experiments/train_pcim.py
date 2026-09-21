"""PCIM training loop.

Every hyperparameter lives in a yaml config (configs/train_pcim_cpu.yaml,
configs/train_pcim_gpu.yaml) - nothing here is hardcoded. Training examples
are built from clean normals via paste_anomaly() + degrade_pair(): each step
samples a clean image, pastes a synthetic anomaly (giving a known mask), then
degrades the with-anomaly and without-anomaly versions with IDENTICAL
parameters and noise (degrade_pair's whole reason to exist) so L_pres is
computable. DBDE estimates the degradation from the without-anomaly member of
the pair (defect-free by construction, so the cleanest signal for a blind
estimator) and the SAME estimate is used for both forward passes, since both
members really did share one degradation.

Resumable: a single rolling checkpoint per run (checkpoints/<run>.pt) holds
model + optimizer state, the step counter, and the numpy Generator's bit-
generator state - not just the model. Restoring the RNG state is what makes
"resume cleanly" mean "continues the same sample sequence," not just "loads
weights and starts sampling from a different point than an uninterrupted run
would have." That property is exactly what ablate_lpres.py leans on for an
apples-to-apples comparison.

Per-step losses go to a CSV (results/<run>_losses.csv), not stdout - only a
compact one-line status prints, every `log_every` steps. Held-out evaluation
(relative DRR - overall AND broken out per anomaly kind, since an aggregate
hides exactly the scratch case that matters most - DRemR, PSNR on normal
regions) runs every `eval_every` steps against a FIXED set of eval examples
built once at startup, logged to a second CSV (results/<run>_eval.csv).

Loss normalisation (loss.normalize in the config, default on): L_freq's raw
magnitude runs 1-2 orders of magnitude above the other three terms (FFT-
magnitude L1 is dominated by a handful of low-frequency bins), which would
otherwise make an "8-hour CPU run" mostly optimise spectral fidelity and
tell you nothing about whether L_pres does anything. On the first real
training batch, each term's raw magnitude is measured and a per-term scale
factor (1/raw) is computed so every term starts at the same order of
magnitude; the configured w_rec/w_deg/w_freq/w_pres weights are applied ON
TOP of that common baseline, so they control genuine relative importance
rather than fighting a magnitude imbalance they didn't cause. Both the raw
and scaled value of every term are logged (rec_raw/rec_scaled/... columns)
so the balance is visible, not just asserted. The computed scale factors
are stored in the checkpoint - a resumed run reuses them rather than
recalibrating from whatever batch it happens to resume on, which would
otherwise make the loss scale (and therefore the effective learning rate
per term) drift depending on where a run was interrupted.

    python -m src.experiments.train_pcim --config configs/train_pcim_cpu.yaml --smoke
    python -m src.experiments.train_pcim --config configs/train_pcim_cpu.yaml
    python -m src.experiments.train_pcim --config configs/train_pcim_gpu.yaml --device cuda
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.mvtec import load_train_normals, synthetic_split
from src.dbde.estimator import DegradationEstimate, estimate, reference_psd
from src.degrade.anomaly import ANOMALY_KINDS, TextureBank, paste_anomaly
from src.degrade.simulator import degrade_pair
from src.metrics.core import (
    defect_residual_correlation, defect_retention_ratio, degradation_removal_ratio, psnr,
)
from src.models.losses import (
    degradation_consistency_loss, frequency_loss, preservation_loss, reconstruction_loss,
)
from src.models.pcim import PCIM, assert_physics_sane_stratified
from src.utils.paths import checkpoints_dir, dtd_root, load_config, results_dir

EPS = 1e-8


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_data_pools(cfg: dict, smoke: bool) -> dict[str, list[np.ndarray]]:
    categories = cfg["data"]["categories"]
    size = cfg["data"]["size"]
    n_train = cfg["data"]["n_train_per_category"]
    pools = {}
    for cat in categories:
        imgs = load_train_normals(cat, size=size, limit=n_train, smoke=smoke)
        if not imgs:
            raise RuntimeError(f"no training normals for category {cat!r}")
        pools[cat] = imgs
    return pools


def build_reference_cache(pools: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """One reference_psd() per category, computed once from its own clean
    training normals. This is what makes reference-based blur detection
    (the primary path in src.dbde.estimator - see its module docstring)
    possible during training: DBDE needs the SAME category's clean spectrum
    to test a query image against, not a generic one."""
    return {cat: reference_psd(imgs) for cat, imgs in pools.items()}


@dataclass
class Example:
    category: str
    x0: np.ndarray            # clean, no anomaly
    x_a: np.ndarray           # clean, with anomaly
    mask: np.ndarray          # (H,W) 0/1
    kind: str
    family: str
    severity: int
    y_a: np.ndarray           # degraded, with anomaly
    y_0: np.ndarray           # degraded, without anomaly (counterfactual)
    est: DegradationEstimate  # DBDE estimate from y_0


def _sample_example(rng: np.random.Generator, pools: dict[str, list[np.ndarray]],
                    cfg: dict, bank: TextureBank,
                    ref_cache: dict[str, np.ndarray] | None = None) -> Example:
    categories = list(pools)
    category = categories[rng.integers(len(categories))]
    pool = pools[category]
    x0 = pool[rng.integers(len(pool))]

    kind = str(rng.choice(cfg["anomaly"]["kinds"]))
    x_a, mask, _spec = paste_anomaly(x0, rng, bank=bank, kind=kind)
    if mask.sum() == 0:
        # degenerate paste (can happen at tiny smoke sizes) - retry once with a blob
        x_a, mask, _spec = paste_anomaly(x0, rng, bank=bank, kind="blob")

    family = str(rng.choice(cfg["degrade"]["families"]))
    sev_lo, sev_hi = cfg["degrade"]["severity_range"]
    severity = int(rng.integers(sev_lo, sev_hi + 1))
    pair_seed = int(rng.integers(0, 2 ** 31 - 1))
    y_a, y_0, _params = degrade_pair(x_a, x0, family, severity, seed=pair_seed)

    ref_logpsd = ref_cache.get(category) if ref_cache else None
    est = estimate(y_0, ref_logpsd=ref_logpsd)

    return Example(category=category, x0=x0, x_a=x_a, mask=mask, kind=kind,
                   family=family, severity=severity, y_a=y_a, y_0=y_0, est=est)


def build_eval_set(cfg: dict, pools: dict[str, list[np.ndarray]], bank: TextureBank,
                   ref_cache: dict[str, np.ndarray] | None = None) -> list[Example]:
    """Fixed, built once at startup from a SEPARATE seed than training - the
    same set is reused at every eval_every checkpoint, so eval numbers are
    comparable across steps."""
    rng = np.random.default_rng(cfg["eval"]["seed"])
    n = cfg["eval"]["n_examples"]
    return [_sample_example(rng, pools, cfg, bank, ref_cache) for _ in range(n)]


# --------------------------------------------------------------------------
# tensor plumbing
# --------------------------------------------------------------------------

def _img_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return t.unsqueeze(0).float().to(device)


def _mask_to_tensor(mask: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float().to(device)


def _est_to_tensors(est: DegradationEstimate, device: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    illum = torch.from_numpy(np.ascontiguousarray(est.illum_field)).unsqueeze(0).unsqueeze(0).float().to(device)
    # None when no blur was detected - PCIM.forward() skips the Wiener data
    # step entirely for None, rather than deconvolving against a fake
    # identity kernel (see pcim.py's forward() docstring for why that
    # distinction matters).
    if est.blur_kind == "none":
        kernel = None
    else:
        kernel = torch.from_numpy(np.ascontiguousarray(est.kernel())).unsqueeze(0).unsqueeze(0).float().to(device)
    sigma = torch.tensor([max(est.noise_sigma, 1e-4)], dtype=torch.float32, device=device)
    return illum, kernel, sigma


def _restore_both(model: PCIM, ex: Example, device: str):
    illum, kernel, sigma = _est_to_tensors(ex.est, device)
    y_a = _img_to_tensor(ex.y_a, device)
    y_0 = _img_to_tensor(ex.y_0, device)
    x_full_a, x_cons_a = model(y_a, illum_field=illum, kernel=kernel, sigma=sigma)
    x_full_0, x_cons_0 = model(y_0, illum_field=illum, kernel=kernel, sigma=sigma)
    return x_full_a, x_cons_a, x_full_0, x_cons_0, illum, kernel, y_a, y_0


# --------------------------------------------------------------------------
# losses / eval
# --------------------------------------------------------------------------

def compute_raw_losses(model: PCIM, ex: Example, device: str) -> dict[str, torch.Tensor]:
    """The four RAW, un-weighted, un-normalised loss components. Combining
    them into a training objective (weighting + normalisation) is a
    separate step - see calibrate_scale_factors() / weighted_total() -
    so the raw magnitudes stay inspectable on their own."""
    x_full_a, _, x_full_0, _, illum, kernel, y_a, y_0 = _restore_both(model, ex, device)
    x_a_t = _img_to_tensor(ex.x_a, device)
    x_0_t = _img_to_tensor(ex.x0, device)
    mask_t = _mask_to_tensor(ex.mask, device)

    l_rec = 0.5 * (reconstruction_loss(x_full_a, x_a_t) + reconstruction_loss(x_full_0, x_0_t))
    l_deg = 0.5 * (degradation_consistency_loss(x_full_a, y_a, illum, kernel)
                   + degradation_consistency_loss(x_full_0, y_0, illum, kernel))
    l_freq = 0.5 * (frequency_loss(x_full_a, x_a_t) + frequency_loss(x_full_0, x_0_t))
    l_pres = preservation_loss(x_full_a, x_full_0, x_a_t, x_0_t, mask_t)

    return dict(rec=l_rec, deg=l_deg, freq=l_freq, pres=l_pres)


LOSS_TERMS = ("rec", "deg", "freq", "pres")


@torch.no_grad()
def calibrate_scale_factors(model: PCIM, examples: list[Example], cfg: dict,
                            device: str) -> dict[str, float]:
    """Measure each term's mean raw magnitude over `examples` (the first
    real training batch) and return per-term scale factors (1/raw) so every
    term starts at the same order of magnitude. Returns all-1.0 (a no-op)
    if loss.normalize is set False in the config."""
    if not bool(cfg["loss"].get("normalize", True)):
        return {k: 1.0 for k in LOSS_TERMS}

    sums = {k: 0.0 for k in LOSS_TERMS}
    for ex in examples:
        raw = compute_raw_losses(model, ex, device)
        for k in LOSS_TERMS:
            sums[k] += float(raw[k]) / len(examples)
    return {k: 1.0 / max(sums[k], EPS) for k in LOSS_TERMS}


def weighted_total(raw: dict[str, torch.Tensor], scale_factors: dict[str, float],
                   cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply scale factors then the configured relative weights. Returns
    (total, scaled) where `scaled` holds each term post-normalisation,
    pre-weighting - what gets logged alongside the raw values."""
    scaled = {k: raw[k] * scale_factors[k] for k in LOSS_TERMS}
    w = cfg["loss"]
    use_lpres = bool(w.get("use_lpres", True))
    total = w["w_rec"] * scaled["rec"] + w["w_deg"] * scaled["deg"] + w["w_freq"] * scaled["freq"]
    total = total + (w["w_pres"] * scaled["pres"] if use_lpres else 0.0 * scaled["pres"])
    return total, scaled


@torch.no_grad()
def evaluate(model: PCIM, eval_set: list[Example], device: str,
            physics_check: bool = True, physics_floor: float = -0.5,
            scale_factors: dict[str, float] | None = None,
            cfg: dict | None = None) -> dict[str, float]:
    """Relative DRR is reported overall AND broken out per anomaly kind -
    scratches are the case that decides this project, and they're exactly
    what an aggregate mean would hide if they behaved worse than blobs/
    texture. rel_drr_by_kind[k] is nan if the eval set has no examples of
    kind k with a finite relative DRR.

    residual_corr_{kind} (defect_residual_correlation, per kind) is
    reported alongside relative_drr_{kind} - DRR is a magnitude ratio and
    cannot by itself distinguish genuine defect preservation from ringing
    at similar energy once it's near or above ~1.0 (see that function's
    docstring), which is exactly the regime a real L_pres effect would
    need to be read in.

    Also evaluates x_cons (the pure-physics, no-learned-component path) and,
    if physics_check is set, raises loudly (assert_physics_sane_stratified)
    when its DRemR fails any of that function's three tiers - x_cons has no
    mechanism to blame a training-driven excuse for being wrong, so this is
    always a physics-chain bug when it fires, not a training issue, and it
    should stop the run rather than be silently absorbed by hours of
    further training on top of it (see README/HANDOFF for exactly this
    happening before tier 1 alone existed, and eval_pcim_holdout.py's
    physics_guard_check for why tier 1 alone has a real blind spot an
    eval-set mean can hide behind).

    blur_kind_* counts come from the FIXED eval set's own DBDE estimates
    (computed once, at build_eval_set() time, not re-estimated here) - they
    won't change step to step unless the eval set itself changes, but
    logging them at every eval row is still a real check: it keeps
    detection behaviour visible in the same trace as the metrics it's
    affecting, instead of something you'd only think to check if a run
    already looked wrong.

    If scale_factors and cfg are given, also computes the four SCALED loss
    terms averaged over the eval set (not a noisy training batch) - the
    same rec/deg/freq/pres_scaled quantities logged every training step,
    but on a fixed set so the balance is comparable across the whole run in
    one file, not split between losses.csv and eval.csv.
    """
    model.eval()
    # drr_methods/drr_ids accumulate RAW per-example values, not per-example
    # ratios - relative DRR is reported as ratio-of-means
    # (mean(drr_method)/mean(drr_id)), never mean-of-per-example-ratios. See
    # relative_drr's docstring in metrics/core.py: mean-of-ratios is
    # dominated by whichever examples have a small drr_id denominator, the
    # same instability DRemR has at small degradation magnitudes.
    drr_methods, drr_ids, dremrs, psnrs = [], [], [], []
    drr_methods_by_kind: dict[str, list[float]] = {k: [] for k in ANOMALY_KINDS}
    drr_ids_by_kind: dict[str, list[float]] = {k: [] for k in ANOMALY_KINDS}
    residual_corrs_by_kind: dict[str, list[float]] = {k: [] for k in ANOMALY_KINDS}
    cons_dremrs, cons_psnrs = [], []
    cons_dremr_records: list[tuple[str, int, float]] = []
    for ex in eval_set:
        x_full_a, x_cons_a, x_full_0, _, _, _, _, _ = _restore_both(model, ex, device)
        r_a = x_full_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        r_0 = x_full_0.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        rc_a = x_cons_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        normal_mask = ~ex.mask.astype(bool)

        drr_id = defect_retention_ratio(ex.y_a, ex.y_0, ex.x_a, ex.x0, ex.mask)
        drr_method = defect_retention_ratio(r_a, r_0, ex.x_a, ex.x0, ex.mask)
        if np.isfinite(drr_method) and np.isfinite(drr_id):
            drr_methods.append(drr_method)
            drr_ids.append(drr_id)
            drr_methods_by_kind.setdefault(ex.kind, []).append(drr_method)
            drr_ids_by_kind.setdefault(ex.kind, []).append(drr_id)

        # Shape check alongside DRR's magnitude check - see
        # defect_residual_correlation's docstring: DRR alone can't tell
        # genuine preservation apart from ringing at similar energy once
        # it's >= ~1.0, which is exactly the regime a real L_pres effect
        # would show up in.
        rcorr = defect_residual_correlation(r_a, r_0, ex.x_a, ex.x0, ex.mask)
        if np.isfinite(rcorr):
            residual_corrs_by_kind.setdefault(ex.kind, []).append(rcorr)

        dremr = degradation_removal_ratio(r_a, ex.y_a, ex.x_a, ex.mask)
        if np.isfinite(dremr):
            dremrs.append(dremr)

        p = psnr(r_a, ex.x_a, mask=normal_mask)
        if np.isfinite(p):
            psnrs.append(p)

        cons_dremr = degradation_removal_ratio(rc_a, ex.y_a, ex.x_a, ex.mask)
        if np.isfinite(cons_dremr):
            cons_dremrs.append(cons_dremr)
            cons_dremr_records.append((ex.family, ex.severity, cons_dremr))
        cons_p = psnr(rc_a, ex.x_a, mask=normal_mask)
        if np.isfinite(cons_p):
            cons_psnrs.append(cons_p)

    model.train()

    def _ratio_of_means(nums: list[float], dens: list[float]) -> float:
        if not nums or not dens:
            return float("nan")
        m_den = float(np.mean(dens))
        if not np.isfinite(m_den) or abs(m_den) <= EPS:
            return float("nan")
        return float(np.mean(nums) / m_den)

    x_cons_dremr = float(np.mean(cons_dremrs)) if cons_dremrs else float("nan")
    out = dict(
        relative_drr=_ratio_of_means(drr_methods, drr_ids),
        dremr=float(np.mean(dremrs)) if dremrs else float("nan"),
        psnr_normal=float(np.mean(psnrs)) if psnrs else float("nan"),
        n=len(eval_set),
        x_cons_dremr=x_cons_dremr,
        x_cons_psnr_normal=float(np.mean(cons_psnrs)) if cons_psnrs else float("nan"),
    )
    for kind in ANOMALY_KINDS:
        out[f"relative_drr_{kind}"] = _ratio_of_means(
            drr_methods_by_kind.get(kind, []), drr_ids_by_kind.get(kind, []))
        vals = residual_corrs_by_kind.get(kind, [])
        out[f"residual_corr_{kind}"] = float(np.mean(vals)) if vals else float("nan")

    blur_counts = Counter(ex.est.blur_kind for ex in eval_set)
    for kind in ("none", "defocus", "motion"):
        out[f"blur_kind_{kind}"] = blur_counts.get(kind, 0)

    if scale_factors is not None and cfg is not None:
        sums = {k: 0.0 for k in LOSS_TERMS}
        for ex in eval_set:
            raw = compute_raw_losses(model, ex, device)
            _, scaled = weighted_total(raw, scale_factors, cfg)
            for k in LOSS_TERMS:
                sums[k] += float(scaled[k].detach()) / len(eval_set)
        for k in LOSS_TERMS:
            out[f"{k}_scaled"] = sums[k]

    if physics_check and cons_dremr_records:
        assert_physics_sane_stratified(cons_dremr_records, aggregate_floor=physics_floor,
                                       cell_floor=physics_floor)

    return out


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def checkpoint_path(run_name: str) -> Path:
    return checkpoints_dir() / f"{run_name}.pt"


def save_checkpoint(run_name: str, model: PCIM, optimizer: torch.optim.Optimizer,
                    step: int, rng: np.random.Generator, cfg: dict,
                    scale_factors: dict[str, float] | None) -> None:
    path = checkpoint_path(run_name)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(dict(
        step=step,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        rng_state=rng.bit_generator.state,
        config=cfg,
        scale_factors=scale_factors,
    ), tmp)
    tmp.replace(path)  # atomic on POSIX and Windows - never leaves a half-written checkpoint


def load_checkpoint(run_name: str, model: PCIM, optimizer: torch.optim.Optimizer,
                    rng: np.random.Generator, device: str) -> tuple[int, dict[str, float] | None]:
    """Returns (resumed step count, scale_factors) - (0, None) if no
    checkpoint exists. Reusing the checkpoint's scale_factors rather than
    recalibrating on resume matters: recalibrating from whatever batch a
    run happens to resume on would make the loss scale (and therefore each
    term's effective learning rate) drift depending on where the run was
    interrupted, rather than being fixed once by the first real batch."""
    path = checkpoint_path(run_name)
    if not path.exists():
        return 0, None
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    rng.bit_generator.state = ckpt["rng_state"]
    return int(ckpt["step"]), ckpt.get("scale_factors")


# --------------------------------------------------------------------------
# CSV logging
# --------------------------------------------------------------------------

class CsvLogger:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        is_new = not path.exists()
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------
# main training loop - shared by the CLI and ablate_lpres.py
# --------------------------------------------------------------------------

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: dict, run_name: str, device: str | None = None,
       resume: bool = True, max_steps: int | None = None,
       smoke: bool = False, quiet: bool = False) -> dict[str, float]:
    """Train PCIM per `cfg`, checkpointing under `run_name`. Returns the
    final evaluate() dict. `max_steps` overrides cfg["train"]["steps"] -
    used by the smoke/resume tests to stop early without editing the config."""
    device = device or default_device()
    seed = int(cfg["seed"])
    torch.manual_seed(seed)  # model init - only matters for a FRESH run

    pools = load_data_pools(cfg, smoke=smoke)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    ref_cache = build_reference_cache(pools)  # one reference_psd() per category

    model_cfg = cfg["model"]
    model = PCIM(n_iters=model_cfg["n_iters"], rho0=model_cfg["rho0"],
                rho_scale=model_cfg["rho_scale"], prox_width=model_cfg["prox_width"],
                prox_depth=model_cfg["prox_depth"], illum_clamp=model_cfg["illum_clamp"],
                vst_gain=model_cfg["vst_gain"],
                nsr_floor=model_cfg.get("nsr_floor", 0.01)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    rng = np.random.default_rng(seed)
    step, scale_factors = load_checkpoint(run_name, model, optimizer, rng, device) if resume else (0, None)

    eval_set = build_eval_set(cfg, pools, bank, ref_cache)  # rebuilt identically every run (own seed)

    total_steps = max_steps if max_steps is not None else cfg["train"]["steps"]
    log_every = cfg["train"]["log_every"]
    ckpt_every = cfg["train"]["checkpoint_every"]
    eval_every = cfg["train"]["eval_every"]
    batch_size = cfg["train"]["batch_size"]
    physics_check = bool(cfg.get("eval", {}).get("physics_sanity_check", True))
    physics_floor = float(cfg.get("eval", {}).get("physics_sanity_dremr_floor", -0.5))

    loss_fields = ["step", "total", "elapsed_s"]
    for k in LOSS_TERMS:
        loss_fields += [f"{k}_raw", f"{k}_scaled"]
    loss_csv = CsvLogger(results_dir() / f"{run_name}_losses.csv", loss_fields)
    eval_fields = ["step", "relative_drr", "dremr", "psnr_normal", "n",
                  "x_cons_dremr", "x_cons_psnr_normal"]
    eval_fields += [f"relative_drr_{k}" for k in ANOMALY_KINDS]
    eval_fields += [f"residual_corr_{k}" for k in ANOMALY_KINDS]
    eval_fields += ["blur_kind_none", "blur_kind_defocus", "blur_kind_motion"]
    eval_fields += [f"{k}_scaled" for k in LOSS_TERMS]
    eval_csv = CsvLogger(results_dir() / f"{run_name}_eval.csv", eval_fields)

    t0 = time.time()
    last_eval = None
    try:
        while step < total_steps:
            optimizer.zero_grad()
            examples = [_sample_example(rng, pools, cfg, bank, ref_cache) for _ in range(batch_size)]

            if scale_factors is None:
                # First real training batch of a fresh run: calibrate once,
                # from exactly the batch about to be trained on. Cheap
                # (no_grad, discarded) - the real forward pass below is
                # what actually gets a gradient.
                scale_factors = calibrate_scale_factors(model, examples, cfg, device)
                if not quiet:
                    print(f"[{run_name}] loss scale factors (calibrated at step {step + 1}): "
                          f"{ {k: round(v, 4) for k, v in scale_factors.items()} }", flush=True)

            sums = {"total": 0.0}
            for k in LOSS_TERMS:
                sums[f"{k}_raw"] = 0.0
                sums[f"{k}_scaled"] = 0.0

            for ex in examples:
                raw = compute_raw_losses(model, ex, device)
                total, scaled = weighted_total(raw, scale_factors, cfg)
                (total / batch_size).backward()
                sums["total"] += float(total.detach()) / batch_size
                for k in LOSS_TERMS:
                    sums[f"{k}_raw"] += float(raw[k].detach()) / batch_size
                    sums[f"{k}_scaled"] += float(scaled[k].detach()) / batch_size

            optimizer.step()
            step += 1

            loss_csv.write(dict(step=step, elapsed_s=round(time.time() - t0, 2), **sums))

            if step % ckpt_every == 0 or step == total_steps:
                save_checkpoint(run_name, model, optimizer, step, rng, cfg, scale_factors)

            if step % eval_every == 0 or step == total_steps:
                last_eval = evaluate(model, eval_set, device,
                                     physics_check=physics_check, physics_floor=physics_floor,
                                     scale_factors=scale_factors, cfg=cfg)
                eval_csv.write(dict(step=step, **last_eval))

            if step % log_every == 0 or step == total_steps:
                if not quiet:
                    msg = f"[{run_name}] step {step}/{total_steps}  total={sums['total']:.4f}"
                    if last_eval is not None:
                        msg += f"  rel_drr={last_eval['relative_drr']:.3f}"
                    print(msg, flush=True)
    finally:
        loss_csv.close()
        eval_csv.close()

    save_checkpoint(run_name, model, optimizer, step, rng, cfg, scale_factors)  # final state, always
    return last_eval if last_eval is not None else evaluate(
        model, eval_set, device, physics_check=physics_check, physics_floor=physics_floor,
        scale_factors=scale_factors, cfg=cfg)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SMOKE_OVERRIDES = dict(
    data=dict(size=32, n_train_per_category=4),
    train=dict(steps=6, batch_size=1, log_every=2, checkpoint_every=2, eval_every=3),
    eval=dict(n_examples=2),
)


def _apply_smoke_overrides(cfg: dict) -> dict:
    for section, overrides in SMOKE_OVERRIDES.items():
        cfg.setdefault(section, {}).update(overrides)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny synthetic run, a handful of steps, under a minute")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = _apply_smoke_overrides(cfg)

    run_name = args.run_name or Path(args.config).stem
    device = args.device or default_device()

    print(f"run_name : {run_name}")
    print(f"device   : {device}")
    print(f"smoke    : {args.smoke}")

    final = run(cfg, run_name, device=device, resume=not args.fresh, smoke=args.smoke)
    print(f"\nfinal eval: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
