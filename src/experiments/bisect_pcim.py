"""Bisect PCIM's physics chain to find exactly which stage breaks it.

diagnose_pcim.py established that x_cons (pure physics, no learned prox) is
already catastrophic - worse than the raw degraded input. This script finds
WHERE in illum-divide -> VST -> Wiener-HQS -> inverse-VST -> illum-restore
the damage happens, by evaluating each prefix of that chain on the same eval
set, plus four targeted unit checks on the individual primitives in
isolation. Read-only: calls the real functions from src.models.pcim, does
not modify them.

    python -m src.experiments.bisect_pcim --config configs/train_pcim_cpu.yaml
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from src.data.mvtec import load_train_normals
from src.dbde.estimator import estimate
from src.degrade.simulator import apply_kernel, defocus_kernel
from src.experiments.train_pcim import (
    _est_to_tensors, _img_to_tensor, build_eval_set, load_data_pools,
)
from src.degrade.anomaly import TextureBank
from src.metrics.core import degradation_removal_ratio, psnr
from src.models.pcim import (
    _kernel_to_otf, generalized_anscombe, identity_otf, illumination_divide,
    illumination_restore, inverse_generalized_anscombe, wiener_data_step,
)
from src.utils.paths import dtd_root, load_config

EPS = 1e-8


# --------------------------------------------------------------------------
# cumulative pipeline bisection
# --------------------------------------------------------------------------

def run_stages(ex, model_cfg: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    illum, kernel, sigma = _est_to_tensors(ex.est, device)
    y = _img_to_tensor(ex.y_a, device)
    b, c, h, w = y.shape
    clamp = model_cfg["illum_clamp"]
    gain = model_cfg["vst_gain"]

    stage0 = y  # identity passthrough

    x1 = illumination_divide(y, illum, clamp)
    stage1 = illumination_restore(x1, illum, clamp)

    x2 = illumination_divide(y, illum, clamp)
    v2 = generalized_anscombe(x2, sigma, gain)
    x2b = inverse_generalized_anscombe(v2, sigma, gain)
    stage2 = illumination_restore(x2b, illum, clamp)

    x3 = illumination_divide(y, illum, clamp)
    v3 = generalized_anscombe(x3, sigma, gain)
    if kernel is not None:
        otf = _kernel_to_otf(kernel, h, w)
    else:
        otf = identity_otf((b, c), h, w, y.dtype, device)
    if otf.shape[1] == 1 and c > 1:
        otf = otf.expand(b, c, h, w)
    xk = v3
    for t in range(model_cfg["n_iters"]):
        rho_t = model_cfg["rho0"] * (model_cfg["rho_scale"] ** t)
        xk = wiener_data_step(v3, xk, otf, sigma, rho_t)
    v3b = inverse_generalized_anscombe(xk, sigma, gain)
    stage3 = illumination_restore(v3b, illum, clamp)

    return dict(stage0=stage0, stage1=stage1, stage2=stage2, stage3=stage3)


@torch.no_grad()
def bisect(cfg: dict, device: str = "cpu") -> dict[str, dict[str, float]]:
    pools = load_data_pools(cfg, smoke=False)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    eval_set = build_eval_set(cfg, pools, bank)

    metrics = {s: {"dremr": [], "psnr": []} for s in
              ("stage0", "stage1", "stage2", "stage3")}

    for ex in eval_set:
        stages = run_stages(ex, cfg["model"], device)
        normal_mask = ~ex.mask.astype(bool)
        for name, t in stages.items():
            r = t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            d = degradation_removal_ratio(r, ex.y_a, ex.x_a, ex.mask)
            if np.isfinite(d):
                metrics[name]["dremr"].append(d)
            p = psnr(r, ex.x_a, mask=normal_mask)
            if np.isfinite(p):
                metrics[name]["psnr"].append(p)

    return {
        name: dict(dremr=float(np.mean(v["dremr"])) if v["dremr"] else float("nan"),
                  psnr=float(np.mean(v["psnr"])) if v["psnr"] else float("nan"),
                  n=len(v["dremr"]))
        for name, v in metrics.items()
    }


def inspect_eval_set_estimates(cfg: dict) -> dict:
    """What DBDE actually estimated on the eval set images that stage 3
    fails on - what is Wiener actually being fed, not a hand-picked nsr."""
    pools = load_data_pools(cfg, smoke=False)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    eval_set = build_eval_set(cfg, pools, bank)

    sigmas, blur_kinds, radii, lengths = [], [], [], []
    for ex in eval_set:
        sigmas.append(ex.est.noise_sigma)
        blur_kinds.append(ex.est.blur_kind)
        if ex.est.blur_kind == "defocus":
            radii.append(ex.est.blur_radius)
        elif ex.est.blur_kind == "motion":
            lengths.append(ex.est.blur_length)

    from collections import Counter
    sigmas = np.array(sigmas)
    return dict(
        n=len(eval_set),
        sigma_min=float(sigmas.min()), sigma_max=float(sigmas.max()),
        sigma_mean=float(sigmas.mean()), sigma_median=float(np.median(sigmas)),
        blur_kind_counts=dict(Counter(blur_kinds)),
        defocus_radii=radii, motion_lengths=lengths,
    )


def check_wiener_actual_pipeline_sigma(img: np.ndarray, true_radius: float,
                                       actual_sigma: float) -> dict:
    """Same single-shot known-kernel setup as check (b), but fed the sigma
    value in the units/scale the REAL pipeline actually uses (the raw DBDE
    noise_sigma, ~0.01-0.06) directly as wiener_data_step's `sigma` argument
    at rho=1 - i.e. exactly how PCIM._hqs's first iteration (rho_t=rho0=1)
    calls it - rather than a hand-picked sqrt(nsr). This is the case
    check (b)'s nsr sweep does NOT directly test."""
    h, w = img.shape[:2]
    k_np = defocus_kernel(true_radius)
    blurred_np = apply_kernel(img, k_np)
    y = torch.from_numpy(blurred_np.transpose(2, 0, 1)).unsqueeze(0).float()
    k_t = torch.from_numpy(k_np).unsqueeze(0).unsqueeze(0).float()
    otf = _kernel_to_otf(k_t, h, w).expand(1, 3, h, w)

    out = {}
    for rho in (1.0, 2.0, 4.0, 8.0, 16.0):  # the real rho0*rho_scale^t schedule
        sigma_param = torch.tensor([actual_sigma])
        x_hat = wiener_data_step(y, y, otf, sigma_param, rho=rho)
        r = x_hat.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
        out[f"rho={rho}"] = dict(effective_reg=rho * actual_sigma ** 2, psnr=psnr(r, img))
    return out


# --------------------------------------------------------------------------
# (a) VST round-trip
# --------------------------------------------------------------------------

def check_vst_roundtrip(img: np.ndarray) -> list[dict]:
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    out = []
    for sigma_val in [0.0, 2 / 255, 5 / 255, 10 / 255, 15 / 255]:
        sigma = torch.tensor([sigma_val])
        v = generalized_anscombe(x, sigma, gain=1.0)
        x_hat = inverse_generalized_anscombe(v, sigma, gain=1.0)
        max_abs_err = float((x_hat - x).abs().max())
        out.append(dict(sigma=sigma_val, max_abs_err=max_abs_err,
                        passes_tol=max_abs_err < 1e-4))
    return out


# --------------------------------------------------------------------------
# (b) Wiener on a known case
# --------------------------------------------------------------------------

def check_wiener_known_kernel(img: np.ndarray, true_radius: float = 3.0) -> dict:
    h, w = img.shape[:2]

    k_np = defocus_kernel(true_radius)
    blurred_np = apply_kernel(img, k_np)  # ground-truth blur, no noise
    y = torch.from_numpy(blurred_np.transpose(2, 0, 1)).unsqueeze(0).float()

    k_t = torch.from_numpy(k_np).unsqueeze(0).unsqueeze(0).float()
    otf_known = _kernel_to_otf(k_t, h, w).expand(1, 3, h, w)

    est = estimate(blurred_np)
    k_est_np = est.kernel()
    k_est_t = torch.from_numpy(k_est_np).unsqueeze(0).unsqueeze(0).float()
    otf_est = _kernel_to_otf(k_est_t, h, w).expand(1, 3, h, w)

    nsr_sweep = [0.001, 0.01, 0.05, 0.2]
    results_known, results_est = [], []
    for nsr in nsr_sweep:
        sigma_param = torch.tensor([np.sqrt(nsr)])
        x_hat_known = wiener_data_step(y, y, otf_known, sigma_param, rho=1.0)
        r = x_hat_known.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
        results_known.append(dict(nsr=nsr, psnr=psnr(r, img)))

        x_hat_est = wiener_data_step(y, y, otf_est, sigma_param, rho=1.0)
        r2 = x_hat_est.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
        results_est.append(dict(nsr=nsr, psnr=psnr(r2, img)))

    return dict(
        true_radius=true_radius,
        estimated_radius=est.blur_radius,
        blurred_vs_clean_psnr=psnr(blurred_np, img),
        known_kernel=results_known,
        estimated_kernel=results_est,
    )


# --------------------------------------------------------------------------
# (c) Wiener phase / spatial shift
# --------------------------------------------------------------------------

def check_wiener_phase(img: np.ndarray, true_radius: float = 3.0, nsr: float = 0.01) -> dict:
    h, w = img.shape[:2]
    k_np = defocus_kernel(true_radius)
    blurred_np = apply_kernel(img, k_np)
    y = torch.from_numpy(blurred_np.transpose(2, 0, 1)).unsqueeze(0).float()
    k_t = torch.from_numpy(k_np).unsqueeze(0).unsqueeze(0).float()
    otf = _kernel_to_otf(k_t, h, w).expand(1, 3, h, w)
    sigma_param = torch.tensor([np.sqrt(nsr)])
    x_hat = wiener_data_step(y, y, otf, sigma_param, rho=1.0)
    out_np = x_hat.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)

    in_g = cv2.cvtColor((np.clip(blurred_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out_g = cv2.cvtColor((np.clip(out_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # FFT-based subpixel phase correlation (cv2.phaseCorrelate), windowed to
    # suppress edge effects - NOT a spatial cross-correlation via filter2D
    # with zero padding, which turned out to give a bogus multi-pixel offset
    # inconsistent with the actual PSNR (a genuine 33x48px shift would wreck
    # PSNR, not improve it over the no-deconv baseline as observed).
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (shift_x, shift_y), response = cv2.phaseCorrelate(in_g * win, out_g * win)

    return dict(offset_y_px=float(shift_y), offset_x_px=float(shift_x),
               phase_corr_response=float(response),
               output_psnr_vs_clean=psnr(out_np, img))


# --------------------------------------------------------------------------
# (d) illumination field / clamp statistics
# --------------------------------------------------------------------------

def check_illumination_stats(images: list[np.ndarray], clamp: float = 4.0) -> dict:
    all_fields = []
    n_clamped = 0
    n_total = 0
    for img in images:
        est = estimate(img)
        field = est.illum_field
        all_fields.append(field.ravel())
        n_clamped += int(np.sum((field <= 1.0 / clamp) | (field >= clamp)))
        n_total += field.size

    all_vals = np.concatenate(all_fields)
    applied_gain = 1.0 / np.clip(all_vals, 1.0 / clamp, clamp)  # what y gets divided by

    return dict(
        field_min=float(all_vals.min()), field_max=float(all_vals.max()),
        field_mean=float(all_vals.mean()),
        field_p1=float(np.percentile(all_vals, 1)), field_p50=float(np.percentile(all_vals, 50)),
        field_p99=float(np.percentile(all_vals, 99)),
        applied_gain_min=float(applied_gain.min()), applied_gain_max=float(applied_gain.max()),
        applied_gain_mean=float(applied_gain.mean()),
        frac_clamped=float(n_clamped / max(n_total, 1)),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref_imgs = load_train_normals("carpet", size=cfg["data"]["size"], limit=6, smoke=False)
    ref_img = ref_imgs[0]

    print("=" * 70)
    print("CUMULATIVE PIPELINE BISECTION")
    print("=" * 70)
    stage_results = bisect(cfg, device=args.device)
    print(f"{'stage':10s} {'DRemR':>10s} {'PSNR (dB)':>10s}  n")
    labels = {"stage0": "0 identity", "stage1": "1 illum", "stage2": "2 +VST",
             "stage3": "3 +Wiener (=x_cons)"}
    for name in ("stage0", "stage1", "stage2", "stage3"):
        r = stage_results[name]
        print(f"{labels[name]:22s} {r['dremr']:>10.3f} {r['psnr']:>10.3f}  {r['n']}")

    print("\n" + "=" * 70)
    print("(a) VST ROUND-TRIP")
    print("=" * 70)
    for row in check_vst_roundtrip(ref_img):
        print(f"  sigma={row['sigma']:.5f}  max_abs_err={row['max_abs_err']:.6e}  "
              f"passes(<1e-4)={row['passes_tol']}")

    print("\n" + "=" * 70)
    print("(b) WIENER ON A KNOWN CASE")
    print("=" * 70)
    wb = check_wiener_known_kernel(ref_img)
    print(f"  true radius={wb['true_radius']}  estimated radius={wb['estimated_radius']:.3f}")
    print(f"  blurred (no deconv) vs clean PSNR: {wb['blurred_vs_clean_psnr']:.3f} dB")
    print("  known kernel:")
    for row in wb["known_kernel"]:
        print(f"    nsr={row['nsr']:<8} PSNR={row['psnr']:.3f} dB")
    print("  DBDE-estimated kernel:")
    for row in wb["estimated_kernel"]:
        print(f"    nsr={row['nsr']:<8} PSNR={row['psnr']:.3f} dB")

    print("\n" + "=" * 70)
    print("(c) WIENER PHASE / SPATIAL SHIFT")
    print("=" * 70)
    wc = check_wiener_phase(ref_img)
    print(f"  offset: ({wc['offset_y_px']:.3f}, {wc['offset_x_px']:.3f}) px  "
          f"(phase-corr response={wc['phase_corr_response']:.3f})  "
          f"output PSNR vs clean: {wc['output_psnr_vs_clean']:.3f} dB")

    print("\n" + "=" * 70)
    print("(d) ILLUMINATION FIELD / CLAMP STATS")
    print("=" * 70)
    wd = check_illumination_stats(ref_imgs, clamp=cfg["model"]["illum_clamp"])
    for k, v in wd.items():
        print(f"  {k:20s} {v}")

    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: what DBDE actually estimates on the failing eval set")
    print("=" * 70)
    est_stats = inspect_eval_set_estimates(cfg)
    print(f"  n={est_stats['n']}  noise_sigma: min={est_stats['sigma_min']:.5f} "
          f"max={est_stats['sigma_max']:.5f} mean={est_stats['sigma_mean']:.5f} "
          f"median={est_stats['sigma_median']:.5f}")
    print(f"  blur_kind counts: {est_stats['blur_kind_counts']}")
    if est_stats["defocus_radii"]:
        print(f"  defocus radii estimated: {[round(r, 2) for r in est_stats['defocus_radii']]}")
    if est_stats["motion_lengths"]:
        print(f"  motion lengths estimated: {[round(l, 2) for l in est_stats['motion_lengths']]}")

    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: Wiener fed the PIPELINE'S ACTUAL sigma (not a hand-picked nsr),")
    print("across the real rho0*rho_scale^t schedule, known kernel, no noise")
    print("=" * 70)
    for sigma_val in (est_stats["sigma_median"], 0.01, 0.05):
        print(f"  actual_sigma={sigma_val:.5f}:")
        wr = check_wiener_actual_pipeline_sigma(ref_img, true_radius=3.0, actual_sigma=sigma_val)
        for rho_label, row in wr.items():
            print(f"    {rho_label:8s} effective_reg(rho*sigma^2)={row['effective_reg']:.6f}  "
                  f"PSNR={row['psnr']:.3f} dB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
