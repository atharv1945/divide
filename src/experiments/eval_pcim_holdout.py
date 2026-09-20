"""Two post-hoc diagnostics on a trained PCIM checkpoint. Both reuse the
FINAL checkpoint rather than reconstructing training history, because
train_pcim.py keeps a single rolling checkpoint by design (see its module
docstring) - the models that produced any one row of results/<run>_eval.csv
no longer exist once training has moved past that step, and reconstructing
them would mean re-running most of training.

(a) HELD-OUT EVAL: evaluates the final checkpoint on a larger, differently-
    seeded eval set (default 150 examples spanning all categories and
    severities) than the 24-example set training evaluated against - a
    single point from a 24-example set is noisy; this gives a trustworthy
    final DRemR/PSNR/relative-DRR-per-kind with a reported std, not a
    single number.

(b) RATIO-INSTABILITY CHECK: tests, cross-sectionally, whether DRemR (a
    RATIO, denominator = metrics.core.degradation_magnitude()) is
    inherently noisier than PSNR (an absolute measure) specifically on
    mildly-degraded examples - a property of the metric, not evidence of
    an unstable model. DRemR's sensitivity to restoration error scales as
    1/degradation_magnitude (exact closed form, see that function's
    docstring), so examples split into thirds by degradation_magnitude
    should show DRemR spread shrinking from the mildest third to the most
    severe third, while PSNR's spread should not show the same pattern
    (its denominator is a fixed data range, not a per-example quantity).
    This needs only ONE evaluation of the final checkpoint over many
    examples, not multiple training-time snapshots, because it compares
    examples against each other rather than the same example across steps.

    Also reports the degradation_magnitude distribution on the EXACT fixed
    24-example set training evaluated against throughout the run. That
    distribution depends only on (degraded, clean) pairs, not on model
    weights, so it's answerable without any historical checkpoint: if a
    few examples have a much smaller degradation_magnitude than the rest,
    they would dominate any aggregate DRemR swing on their own (a "few
    examples swinging" pattern), independent of what the model was doing
    at any particular step.

    python -m src.experiments.eval_pcim_holdout --config configs/train_pcim_cpu.yaml --run-name train_pcim_cpu
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from src.degrade.anomaly import ANOMALY_KINDS, TextureBank
from src.experiments.train_pcim import (
    _restore_both, build_eval_set, build_reference_cache, checkpoint_path,
    load_data_pools,
)
from src.metrics.core import (
    defect_retention_ratio, degradation_magnitude, degradation_removal_ratio,
    psnr, relative_drr,
)
from src.models.pcim import PCIM
from src.utils.paths import dtd_root, load_config


def load_model(cfg: dict, run_name: str, device: str) -> PCIM:
    ckpt = torch.load(checkpoint_path(run_name), map_location=device)
    m = cfg["model"]
    model = PCIM(n_iters=m["n_iters"], rho0=m["rho0"], rho_scale=m["rho_scale"],
                prox_width=m["prox_width"], prox_depth=m["prox_depth"],
                illum_clamp=m["illum_clamp"], vst_gain=m["vst_gain"],
                nsr_floor=m.get("nsr_floor", 0.01)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def build_holdout_eval_set(cfg: dict, pools, bank, ref_cache, seed: int, n: int):
    """Same construction as train_pcim.build_eval_set, but with an
    overridden seed/size so it draws a set disjoint in practice from the
    one training evaluated against, without mutating the caller's cfg."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["eval"]["seed"] = seed
    cfg2["eval"]["n_examples"] = n
    return build_eval_set(cfg2, pools, bank, ref_cache)


@torch.no_grad()
def per_example_eval(model: PCIM, eval_set, device: str) -> list[dict]:
    """One row per example, carrying everything both diagnostics need:
    den (degradation_magnitude, DRemR's denominator), dremr, psnr, relative
    DRR, plus category/kind/family/severity for breakdowns."""
    rows = []
    for ex in eval_set:
        x_full_a, _, x_full_0, _, _, _, _, _ = _restore_both(model, ex, device)
        r_a = x_full_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        r_0 = x_full_0.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        normal_mask = ~ex.mask.astype(bool)

        drr_id = defect_retention_ratio(ex.y_a, ex.y_0, ex.x_a, ex.x0, ex.mask)
        drr_method = defect_retention_ratio(r_a, r_0, ex.x_a, ex.x0, ex.mask)
        rel = relative_drr(drr_method, drr_id)

        den = degradation_magnitude(ex.y_a, ex.x_a, ex.mask)
        dremr = degradation_removal_ratio(r_a, ex.y_a, ex.x_a, ex.mask)
        p = psnr(r_a, ex.x_a, mask=normal_mask)

        rows.append(dict(category=ex.category, kind=ex.kind, family=ex.family,
                         severity=ex.severity, den=den, dremr=dremr, psnr=p,
                         relative_drr=rel))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Task (a): trustworthy final numbers - mean +/- std, not a single
    noisy point."""
    dremrs = np.array([r["dremr"] for r in rows if np.isfinite(r["dremr"])])
    psnrs = np.array([r["psnr"] for r in rows if np.isfinite(r["psnr"])])
    out = dict(
        n=len(rows),
        dremr_mean=float(dremrs.mean()) if len(dremrs) else float("nan"),
        dremr_std=float(dremrs.std()) if len(dremrs) else float("nan"),
        psnr_mean=float(psnrs.mean()) if len(psnrs) else float("nan"),
        psnr_std=float(psnrs.std()) if len(psnrs) else float("nan"),
    )
    for kind in ANOMALY_KINDS:
        vals = [r["relative_drr"] for r in rows
               if r["kind"] == kind and np.isfinite(r["relative_drr"])]
        out[f"relative_drr_{kind}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def ratio_instability_check(rows: list[dict]) -> dict:
    """Task (b), cross-sectional half: split examples into thirds by
    degradation_magnitude (DRemR's denominator) and compare DRemR spread in
    the mildest third against the most severe third. The ratio-instability
    hypothesis predicts the mildest third's DRemR std should be
    substantially larger; PSNR std should NOT show the same pattern."""
    finite = [r for r in rows
             if np.isfinite(r["den"]) and np.isfinite(r["dremr"]) and np.isfinite(r["psnr"])]
    finite.sort(key=lambda r: r["den"])
    n = len(finite)
    third = max(n // 3, 1)
    low, high = finite[:third], finite[-third:]

    def _std(vals):
        return float(np.std(vals)) if vals else float("nan")

    corr_den_dremr = corr_den_psnr = float("nan")
    if n >= 3:
        dens = [r["den"] for r in finite]
        corr_den_dremr = float(np.corrcoef(dens, [r["dremr"] for r in finite])[0, 1])
        corr_den_psnr = float(np.corrcoef(dens, [r["psnr"] for r in finite])[0, 1])

    return dict(
        n=n,
        low_den_range=(min(r["den"] for r in low), max(r["den"] for r in low)) if low else (float("nan"),) * 2,
        high_den_range=(min(r["den"] for r in high), max(r["den"] for r in high)) if high else (float("nan"),) * 2,
        low_den_dremr_std=_std([r["dremr"] for r in low]),
        high_den_dremr_std=_std([r["dremr"] for r in high]),
        low_den_psnr_std=_std([r["psnr"] for r in low]),
        high_den_psnr_std=_std([r["psnr"] for r in high]),
        corr_den_dremr=corr_den_dremr,
        corr_den_psnr=corr_den_psnr,
    )


def original_eval_den_distribution(cfg: dict, pools, bank, ref_cache) -> dict:
    """Task (b), 'few examples vs uniform shift' half: degradation_magnitude
    on the EXACT fixed 24-example set training evaluated against. Depends
    only on (degraded, clean) pairs, not on model weights, so it's
    answerable without any historical checkpoint."""
    eval_set = build_eval_set(cfg, pools, bank, ref_cache)
    dens = np.array([degradation_magnitude(ex.y_a, ex.x_a, ex.mask) for ex in eval_set])
    order = np.argsort(dens)
    med = float(np.median(dens))
    return dict(
        n=len(dens), min=float(dens.min()), median=med, max=float(dens.max()),
        std=float(dens.std()),
        smallest_3=[float(dens[i]) for i in order[:3]],
        ratio_median_to_min=float(med / dens.min()) if dens.min() > 0 else float("inf"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=999_999)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = load_model(cfg, args.run_name, args.device)
    pools = load_data_pools(cfg, smoke=args.smoke)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    ref_cache = build_reference_cache(pools)

    eval_set = build_holdout_eval_set(cfg, pools, bank, ref_cache, args.seed, args.n)
    rows = per_example_eval(model, eval_set, args.device)

    summ = summarize(rows)
    print(f"(a) held-out eval: n={summ['n']} (seed={args.seed}, disjoint from training's eval.seed)")
    print(f"    dremr : {summ['dremr_mean']:.4f} +/- {summ['dremr_std']:.4f}")
    print(f"    psnr  : {summ['psnr_mean']:.3f} +/- {summ['psnr_std']:.3f} dB")
    for kind in ANOMALY_KINDS:
        print(f"    relative_drr[{kind}]: {summ[f'relative_drr_{kind}']:.4f}")
    print()

    check = ratio_instability_check(rows)
    print(f"(b) ratio-instability check (cross-sectional, n={check['n']}):")
    print(f"    mildest third  : den in {check['low_den_range']}, "
         f"dremr_std={check['low_den_dremr_std']:.4f}, psnr_std={check['low_den_psnr_std']:.3f}")
    print(f"    severest third : den in {check['high_den_range']}, "
         f"dremr_std={check['high_den_dremr_std']:.4f}, psnr_std={check['high_den_psnr_std']:.3f}")
    print(f"    corr(den, dremr) = {check['corr_den_dremr']:.4f}")
    print(f"    corr(den, psnr)  = {check['corr_den_psnr']:.4f}")
    print()

    dist = original_eval_den_distribution(cfg, pools, bank, ref_cache)
    print(f"(b) degradation_magnitude distribution on the ORIGINAL 24-example training eval set:")
    print(f"    min={dist['min']:.2f} median={dist['median']:.2f} max={dist['max']:.2f} std={dist['std']:.2f}")
    print(f"    smallest 3: {[round(v, 2) for v in dist['smallest_3']]}")
    print(f"    median/min ratio: {dist['ratio_median_to_min']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
