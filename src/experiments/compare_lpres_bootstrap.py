"""Bootstrap-CI comparison of the two L_pres-ablation checkpoints
(checkpoints/ablate_lpres_on.pt, ablate_lpres_off.pt) on the 150-example
held-out set - inference only, no retraining.

Why: the ablation's own eval set is 24 examples, so each per-kind number in
its headline report rests on roughly eight examples - not enough to trust a
per-kind claim (e.g. "L_pres helps blob more than texture") without a
confidence interval. This re-evaluates both already-trained checkpoints on
the same 150-example held-out set eval_pcim_holdout.py uses, and reports a
95% bootstrap CI on the ON-OFF delta for relative DRR (ratio-of-means, see
metrics.core.relative_drr's docstring), defect_residual_correlation, and
PSNR, per anomaly kind and overall.

The bootstrap is PAIRED: each resample draws example INDICES once and
applies the same draw to both checkpoints' rows, since both were evaluated
on the identical 150 examples - this is the correct resampling unit for a
paired before/after comparison, not two independent resamples.

Read the result as: if a delta's CI excludes zero, the effect is
established at this sample size. If the CI straddles zero, report the
effect as suggestive, not established - do not treat the point estimate
alone as a headline number.

    python -m src.experiments.compare_lpres_bootstrap --config configs/train_pcim_cpu.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.degrade.anomaly import ANOMALY_KINDS, TextureBank
from src.experiments.eval_pcim_holdout import build_holdout_eval_set, load_model, per_example_eval
from src.experiments.train_pcim import build_reference_cache, load_data_pools
from src.utils.paths import dtd_root, load_config, results_dir

EPS = 1e-8


def _ratio_of_means(rows: list[dict]) -> float:
    """Relative DRR aggregated as ratio-of-means over `rows` - see
    metrics.core.relative_drr's docstring for why, never mean-of-per-
    example-ratios."""
    finite = [r for r in rows if np.isfinite(r["drr_method"]) and np.isfinite(r["drr_id"])]
    if not finite:
        return float("nan")
    den = float(np.mean([r["drr_id"] for r in finite]))
    if not np.isfinite(den) or abs(den) <= EPS:
        return float("nan")
    return float(np.mean([r["drr_method"] for r in finite]) / den)


def _mean_field(rows: list[dict], field: str) -> float:
    vals = [r[field] for r in rows if np.isfinite(r[field])]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_delta(on_rows: list[dict], off_rows: list[dict], stat_fn,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap on the ON-OFF delta of stat_fn. on_rows[i] and
    off_rows[i] must be the SAME example (same eval set, same order) -
    each resample draws indices once and applies them to both, preserving
    the pairing rather than resampling ON and OFF independently."""
    n = len(on_rows)
    if n == 0 or n != len(off_rows):
        return dict(point=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                   excludes_zero=False, n_boot=n_boot, n=n)
    rng = np.random.default_rng(seed)
    point = stat_fn(on_rows) - stat_fn(off_rows)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        on_s = [on_rows[i] for i in idx]
        off_s = [off_rows[i] for i in idx]
        deltas[b] = stat_fn(on_s) - stat_fn(off_s)
    lo, hi = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return dict(point=float(point), ci_lo=lo, ci_hi=hi,
               excludes_zero=bool(lo > 0 or hi < 0), n_boot=n_boot, n=n)


def compare(cfg: dict, on_run_name: str, off_run_name: str, device: str,
           seed: int, n: int, n_boot: int, smoke: bool = False) -> dict:
    pools = load_data_pools(cfg, smoke=smoke)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    ref_cache = build_reference_cache(pools)
    eval_set = build_holdout_eval_set(cfg, pools, bank, ref_cache, seed, n)

    model_on = load_model(cfg, on_run_name, device)
    model_off = load_model(cfg, off_run_name, device)
    rows_on = per_example_eval(model_on, eval_set, device)
    rows_off = per_example_eval(model_off, eval_set, device)

    groups = ["overall"] + list(ANOMALY_KINDS)
    out = {}
    for kind in groups:
        on_g = rows_on if kind == "overall" else [r for r in rows_on if r["kind"] == kind]
        off_g = rows_off if kind == "overall" else [r for r in rows_off if r["kind"] == kind]

        drr_ci = bootstrap_delta(on_g, off_g, _ratio_of_means, n_boot=n_boot, seed=0)
        corr_ci = bootstrap_delta(on_g, off_g, lambda r: _mean_field(r, "residual_corr"),
                                  n_boot=n_boot, seed=1)
        psnr_ci = bootstrap_delta(on_g, off_g, lambda r: _mean_field(r, "psnr"),
                                  n_boot=n_boot, seed=2)

        out[kind] = dict(
            n=len(on_g),
            relative_drr_on=_ratio_of_means(on_g), relative_drr_off=_ratio_of_means(off_g),
            relative_drr_delta=drr_ci,
            residual_corr_on=_mean_field(on_g, "residual_corr"),
            residual_corr_off=_mean_field(off_g, "residual_corr"),
            residual_corr_delta=corr_ci,
            psnr_on=_mean_field(on_g, "psnr"), psnr_off=_mean_field(off_g, "psnr"),
            psnr_delta=psnr_ci,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    # ablate_lpres.py's run() names checkpoints "<run_prefix>_lpres_on"/
    # "<run_prefix>_lpres_off" - with the default run_prefix "ablate_lpres"
    # that's "ablate_lpres_lpres_on"/"_lpres_off" (checked against the
    # actual files in checkpoints/, not assumed).
    ap.add_argument("--on-run-name", default="ablate_lpres_lpres_on")
    ap.add_argument("--off-run-name", default="ablate_lpres_lpres_off")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=999_999)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="ablate_lpres_bootstrap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = compare(cfg, args.on_run_name, args.off_run_name, args.device,
                     args.seed, args.n, args.n_boot, smoke=args.smoke)

    print(f"n_boot: {args.n_boot}, eval set: {args.n} examples, seed {args.seed}\n")
    header = (f"{'kind':10s} {'n':>4s}  {'DRR on':>7s} {'DRR off':>8s} "
             f"{'Δdrr':>8s} {'95% CI':>18s}  {'corr on':>7s} {'corr off':>8s} "
             f"{'Δcorr':>7s} {'95% CI':>18s}  {'PSNR Δ':>7s}")
    print(header)
    for kind, r in result.items():
        d, c, p = r["relative_drr_delta"], r["residual_corr_delta"], r["psnr_delta"]
        flag = "" if c["excludes_zero"] else "  (straddles 0)"
        print(f"{kind:10s} {r['n']:>4d}  {r['relative_drr_on']:>7.3f} {r['relative_drr_off']:>8.3f} "
             f"{d['point']:>+8.3f} [{d['ci_lo']:>+.3f},{d['ci_hi']:>+.3f}]  "
             f"{r['residual_corr_on']:>7.3f} {r['residual_corr_off']:>8.3f} "
             f"{c['point']:>+7.3f} [{c['ci_lo']:>+.3f},{c['ci_hi']:>+.3f}]  "
             f"{p['point']:>+7.3f}{flag}")

    out_path = results_dir() / f"{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nresult: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
