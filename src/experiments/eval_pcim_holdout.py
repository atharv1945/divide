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
    defect_residual_correlation, defect_retention_ratio,
    degradation_magnitude, degradation_removal_ratio, psnr,
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
    den (degradation_magnitude, DRemR's denominator), dremr/psnr for x_full
    AND the degraded-input and x_cons baselines (so a bad x_full cell can be
    told apart from "the input was already this bad" vs "the model made it
    worse"), relative DRR, blur-detection diagnostics (blur_kind, the
    estimated kernel size, whether the Wiener step ran at all), plus
    category/kind/family/severity for breakdowns."""
    rows = []
    for ex in eval_set:
        x_full_a, x_cons_a, x_full_0, _, _, _, _, _ = _restore_both(model, ex, device)
        r_a = x_full_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        r_0 = x_full_0.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        rc_a = x_cons_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        normal_mask = ~ex.mask.astype(bool)

        drr_id = defect_retention_ratio(ex.y_a, ex.y_0, ex.x_a, ex.x0, ex.mask)
        drr_method = defect_retention_ratio(r_a, r_0, ex.x_a, ex.x0, ex.mask)
        residual_corr = defect_residual_correlation(r_a, r_0, ex.x_a, ex.x0, ex.mask)

        den = degradation_magnitude(ex.y_a, ex.x_a, ex.mask)
        dremr = degradation_removal_ratio(r_a, ex.y_a, ex.x_a, ex.mask)
        p = psnr(r_a, ex.x_a, mask=normal_mask)
        dremr_cons = degradation_removal_ratio(rc_a, ex.y_a, ex.x_a, ex.mask)
        psnr_cons = psnr(rc_a, ex.x_a, mask=normal_mask)
        # degraded-vs-itself is 0.0 by construction (see
        # degradation_removal_ratio's docstring) - kept anyway so every row
        # carries a like-for-like "did nothing" baseline next to psnr_deg.
        dremr_deg = degradation_removal_ratio(ex.y_a, ex.y_a, ex.x_a, ex.mask)
        psnr_deg = psnr(ex.y_a, ex.x_a, mask=normal_mask)

        est = ex.est
        wiener_ran = est.blur_kind != "none"
        kernel_size = (est.blur_radius if est.blur_kind == "defocus"
                      else est.blur_length if est.blur_kind == "motion" else 0.0)

        rows.append(dict(category=ex.category, kind=ex.kind, family=ex.family,
                         severity=ex.severity, den=den,
                         dremr=dremr, psnr=p,
                         dremr_cons=dremr_cons, psnr_cons=psnr_cons,
                         dremr_deg=dremr_deg, psnr_deg=psnr_deg,
                         # raw ingredients for ratio-of-means relative DRR,
                         # not a per-example ratio - see relative_drr's
                         # docstring in metrics/core.py and summarize()
                         # below, which is the only place these combine.
                         drr_method=drr_method, drr_id=drr_id,
                         residual_corr=residual_corr,
                         blur_kind=est.blur_kind, kernel_size=kernel_size,
                         wiener_ran=wiener_ran))
    return rows


def breakdown_by_family_severity(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Per (family, severity) cell: n, and mean+std of degraded/x_cons/x_full
    DRemR and PSNR - lets a catastrophic cell be told apart as "the input
    was already destroyed" (degraded and x_full both bad) vs "the model
    destroyed it" (degraded fine, x_full bad)."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["severity"]), []).append(r)

    def _stat(grp: list[dict], field: str) -> tuple[float, float]:
        vals = np.array([r[field] for r in grp if np.isfinite(r[field])])
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std())

    out = {}
    for key, grp in groups.items():
        out[key] = dict(
            n=len(grp),
            deg_psnr=_stat(grp, "psnr_deg"), deg_dremr=_stat(grp, "dremr_deg"),
            cons_psnr=_stat(grp, "psnr_cons"), cons_dremr=_stat(grp, "dremr_cons"),
            full_psnr=_stat(grp, "psnr"), full_dremr=_stat(grp, "dremr"),
        )
    return out


def physics_guard_check(rows: list[dict], floor: float = -0.5) -> dict:
    """assert_physics_sane (pcim.py) checks the EVAL-SET MEAN of x_cons's
    DRemR against `floor`. This checks each individual example's x_cons
    DRemR against the same floor, so the two can be compared: a mean well
    above floor can still hide individual examples far below it, if the
    eval set mixes easy and catastrophic cases (exactly what mixing all
    severities in one set does)."""
    finite = [r for r in rows if np.isfinite(r["dremr_cons"])]
    cons_vals = np.array([r["dremr_cons"] for r in finite])
    violations = [r for r in finite if r["dremr_cons"] < floor]
    return dict(
        n=len(finite),
        mean=float(cons_vals.mean()) if len(cons_vals) else float("nan"),
        mean_would_fire=bool(len(cons_vals) and cons_vals.mean() < floor),
        n_individual_violations=len(violations),
        individual_violations=[
            dict(category=r["category"], family=r["family"], severity=r["severity"],
                dremr_cons=r["dremr_cons"], psnr_cons=r["psnr_cons"],
                blur_kind=r["blur_kind"], kernel_size=r["kernel_size"],
                wiener_ran=r["wiener_ran"])
            for r in violations
        ],
    )


def summarize(rows: list[dict]) -> dict:
    """Task (a): trustworthy final numbers - mean +/- std, not a single
    noisy point. Includes the degraded-input and x_cons baselines alongside
    x_full so a bad aggregate can be told apart as "the inputs were already
    this bad" vs "the model made it worse"."""
    def _agg(field):
        vals = np.array([r[field] for r in rows if np.isfinite(r[field])])
        return (float(vals.mean()), float(vals.std())) if len(vals) else (float("nan"), float("nan"))

    dremr_mean, dremr_std = _agg("dremr")
    psnr_mean, psnr_std = _agg("psnr")
    out = dict(
        n=len(rows),
        dremr_mean=dremr_mean, dremr_std=dremr_std,
        psnr_mean=psnr_mean, psnr_std=psnr_std,
        deg_dremr_mean=_agg("dremr_deg")[0], deg_dremr_std=_agg("dremr_deg")[1],
        deg_psnr_mean=_agg("psnr_deg")[0], deg_psnr_std=_agg("psnr_deg")[1],
        cons_dremr_mean=_agg("dremr_cons")[0], cons_dremr_std=_agg("dremr_cons")[1],
        cons_psnr_mean=_agg("psnr_cons")[0], cons_psnr_std=_agg("psnr_cons")[1],
    )
    for kind in ANOMALY_KINDS:
        kind_rows = [r for r in rows if r["kind"] == kind
                    and np.isfinite(r["drr_method"]) and np.isfinite(r["drr_id"])]
        out[f"relative_drr_{kind}"] = _ratio_of_means_rows(kind_rows)
    return out


def _ratio_of_means_rows(rows: list[dict]) -> float:
    """Ratio-of-means relative DRR over a list of per-example row dicts
    (drr_method/drr_id) - see relative_drr's docstring in metrics/core.py.
    Never average a per-example ratio; this is the one aggregation this
    module uses."""
    finite = [r for r in rows if np.isfinite(r["drr_method"]) and np.isfinite(r["drr_id"])]
    if not finite:
        return float("nan")
    den = float(np.mean([r["drr_id"] for r in finite]))
    if not np.isfinite(den) or abs(den) <= 1e-8:
        return float("nan")
    return float(np.mean([r["drr_method"] for r in finite]) / den)


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
    print(f"    {'':10s} {'DRemR':>16s} {'PSNR (dB)':>16s}")
    print(f"    {'degraded':10s} {summ['deg_dremr_mean']:>7.4f} +/- {summ['deg_dremr_std']:<6.4f} "
         f"{summ['deg_psnr_mean']:>7.3f} +/- {summ['deg_psnr_std']:<6.3f}")
    print(f"    {'x_cons':10s} {summ['cons_dremr_mean']:>7.4f} +/- {summ['cons_dremr_std']:<6.4f} "
         f"{summ['cons_psnr_mean']:>7.3f} +/- {summ['cons_psnr_std']:<6.3f}")
    print(f"    {'x_full':10s} {summ['dremr_mean']:>7.4f} +/- {summ['dremr_std']:<6.4f} "
         f"{summ['psnr_mean']:>7.3f} +/- {summ['psnr_std']:<6.3f}")
    for kind in ANOMALY_KINDS:
        print(f"    relative_drr[{kind}]: {summ[f'relative_drr_{kind}']:.4f}")
    print()

    print("(a) breakdown by family x severity (n, degraded/x_cons/x_full PSNR dB, x_full DRemR):")
    breakdown = breakdown_by_family_severity(rows)
    for (family, severity), cell in sorted(breakdown.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"    {family:12s} sev={severity}  n={cell['n']:3d}  "
             f"deg_psnr={cell['deg_psnr'][0]:6.2f}  cons_psnr={cell['cons_psnr'][0]:6.2f}  "
             f"full_psnr={cell['full_psnr'][0]:6.2f}  full_dremr={cell['full_dremr'][0]:+.3f}")
    print()

    guard150 = physics_guard_check(rows, floor=-0.5)
    print(f"(physics guard) per-example x_cons DRemR < -0.5, on the 150-example held-out set:")
    print(f"    mean x_cons DRemR = {guard150['mean']:.4f} "
         f"(aggregate-mean guard would fire: {guard150['mean_would_fire']})")
    print(f"    individual violations: {guard150['n_individual_violations']} / {guard150['n']}")
    for v in guard150["individual_violations"]:
        print(f"      {v['category']:8s} {v['family']:12s} sev={v['severity']}  "
             f"x_cons_dremr={v['dremr_cons']:+.3f}  x_cons_psnr={v['psnr_cons']:6.2f}  "
             f"blur_kind={v['blur_kind']:8s} kernel_size={v['kernel_size']:.2f}  "
             f"wiener_ran={v['wiener_ran']}")
    print()

    orig_eval_set = build_eval_set(cfg, pools, bank, ref_cache)
    orig_rows = per_example_eval(model, orig_eval_set, args.device)
    guard24 = physics_guard_check(orig_rows, floor=-0.5)
    print(f"(physics guard) same check on the ACTUAL 24-example training eval set:")
    print(f"    mean x_cons DRemR = {guard24['mean']:.4f} "
         f"(aggregate-mean guard would fire: {guard24['mean_would_fire']})")
    print(f"    individual violations: {guard24['n_individual_violations']} / {guard24['n']}")
    for v in guard24["individual_violations"]:
        print(f"      {v['category']:8s} {v['family']:12s} sev={v['severity']}  "
             f"x_cons_dremr={v['dremr_cons']:+.3f}  x_cons_psnr={v['psnr_cons']:6.2f}  "
             f"blur_kind={v['blur_kind']:8s} kernel_size={v['kernel_size']:.2f}  "
             f"wiener_ran={v['wiener_ran']}")
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
