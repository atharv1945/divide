"""Sweeps classical Wiener deconvolution's regularization constant (nsr)
from under- to over-regularized and measures scratch relative DRR at each
point - the second half of the bridging-ablation request: if defect
erosion is a continuous, controllable function of regularization strength
rather than a fixed property of "deconvolution" as a method, scratch DRR
should rise monotonically (or close to it) as nsr increases from near-zero
(unregularized, hallucinates high-frequency content near the kernel's
spectral nulls) toward heavily over-regularized (barely deconvolves at
all, converging toward the degraded input).

Same scope as drr_study_deblur (defocus + motion, severities 2-4, 3
categories, 10 images, all anomaly kinds) so this is directly comparable
to that run's "wiener" row (nsr=0.01, scratch 0.093) and to
wiener_bridge_ablation.py's five-configuration table.

    python -m src.experiments.wiener_regularization_sweep
"""
from __future__ import annotations

import functools
import sys

import numpy as np
import pandas as pd

from src.data.mvtec import DEFAULT_CATEGORIES, dataset_available
from src.degrade.anomaly import ANOMALY_KINDS
from src.experiments.drr_study import add_relative_drr, run
from src.models.restorers import CLASSICAL, wiener_deconv
from src.utils.paths import figures_dir, results_dir

NSR_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _register_sweep_restorers() -> list[str]:
    names = []
    for nsr in NSR_VALUES:
        name = f"wiener_nsr_{nsr:g}"
        CLASSICAL[name] = functools.partial(wiener_deconv, nsr=nsr)
        names.append(name)
    return names


def main() -> int:
    if not dataset_available():
        print("MVTec AD not found - this sweep needs real data (set DIVIDE_DATA_ROOT).",
              file=sys.stderr)
        return 2

    names = _register_sweep_restorers()
    restorers = ["identity"] + names

    csv_path = run(
        categories=DEFAULT_CATEGORIES, restorers=restorers,
        families=["defocus", "motion"], severities=[2, 3, 4],
        kinds=ANOMALY_KINDS, n_images=10, size=256, smoke=False, seed=0,
        tag="wiener_nsr_sweep",
    )
    df = add_relative_drr(pd.read_csv(csv_path))
    if df.empty:
        print("no results produced", file=sys.stderr)
        return 1

    rows = []
    for nsr, name in zip(NSR_VALUES, names):
        sub = df[df.restorer == name]
        scratch = sub[sub.anomaly_kind == "scratch"]
        rows.append(dict(
            nsr=nsr,
            drr_rel_mean=float(sub.drr_rel.mean()),
            drr_rel_scratch=float(scratch.drr_rel.mean()) if not scratch.empty else float("nan"),
            dremr_mean=float(sub.dremr.mean()),
            psnr_normal=float(sub.psnr_normal.mean()),
        ))
    sweep = pd.DataFrame(rows)
    sweep_path = results_dir() / "wiener_nsr_sweep_summary.csv"
    sweep.to_csv(sweep_path, index=False)

    print("\n" + "=" * 74)
    print("SCRATCH RELATIVE DRR vs NSR (Wiener regularization constant)")
    print("=" * 74)
    print(sweep.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax1.plot(sweep.nsr, sweep.drr_rel_scratch, "o-", color="tab:red", label="scratch relative DRR")
    ax1.plot(sweep.nsr, sweep.drr_rel_mean, "s--", color="tab:orange", alpha=0.6, label="mean relative DRR (all kinds)")
    ax1.set_xscale("log")
    ax1.axhline(1.0, ls=":", c="green", lw=1.0)
    ax1.axhline(0.5, ls=":", c="gray", lw=1.0)
    ax1.set_xlabel("nsr (Wiener regularization constant, log scale)")
    ax1.set_ylabel("relative DRR")
    ax1.set_title("Defect erosion vs. Wiener regularization strength\n(defocus+motion, severities 2-4, real MVTec)")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig_path = figures_dir() / "wiener_nsr_sweep.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"\ncsv     : {sweep_path}")
    print(f"figure  : {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
