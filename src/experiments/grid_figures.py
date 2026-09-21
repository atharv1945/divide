"""Figures for Step 3's detection grid: gap_closed with bootstrap CIs,
clean/degraded/method AUROC, and the companion scratch DRR + residual
correlation comparison. Reads only files already on disk (step3_grid_
summary.csv, step3_grid_gapclosed_bootstrap.csv, step3_drr_companion_by_
kind.csv) - no re-running of the grid or the bootstrap.

    python -m src.experiments.grid_figures
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import figures_dir, results_dir

RESTORER_ORDER = ["identity", "wiener", "restormer_deblur", "divide_lpres_on", "divide_lpres_off"]
COLORS = {  # colorblind-safe (Okabe-Ito)
    "identity": "#999999", "wiener": "#E69F00", "restormer_deblur": "#56B4E9",
    "divide_lpres_on": "#009E73", "divide_lpres_off": "#D55E00",
}


def make_figures(out: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or figures_dir()
    paths = []

    boot = pd.read_csv(results_dir() / "step3_grid_gapclosed_bootstrap.csv")
    summary = pd.read_csv(results_dir() / "step3_grid_summary.csv")

    # --- forest plot: gap_closed with 95% bootstrap CI, resampling test images ---
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=150)
    detectors = sorted(boot.detector.unique())
    y = 0
    yticks, ylabels = [], []
    for detector in detectors:
        for restorer in RESTORER_ORDER:
            row = boot[(boot.detector == detector) & (boot.restorer == restorer)]
            if row.empty:
                continue
            r = row.iloc[0]
            established = bool(r.excludes_zero)
            color = COLORS.get(restorer, "black")
            ax.errorbar(r.gap_closed_point, y,
                       xerr=[[r.gap_closed_point - r.ci_lo], [r.ci_hi - r.gap_closed_point]],
                       fmt="o", color=color, ecolor=color,
                       markerfacecolor=color if established else "white",
                       markeredgecolor=color, capsize=3, zorder=3)
            yticks.append(y); ylabels.append(f"{restorer} / {detector}")
            y -= 1
        y -= 0.6
    ax.axvline(0.0, ls="--", c="black", lw=1.0, alpha=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("gap_closed  (ratio-of-means; 0 = no better than degraded, 1 = fully recovered clean AUROC)")
    ax.set_title("Detection gap_closed by restorer x detector\n"
                 "(filled = 95% CI excludes zero; open = not established at n_test=16)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    p = out / "step3_gapclosed_forest.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- clean / degraded / method AUROC by restorer x detector ---
    fig, axes = plt.subplots(1, len(detectors), figsize=(6.2 * len(detectors), 4.6), dpi=150, sharey=True)
    if len(detectors) == 1:
        axes = [axes]
    for ax, detector in zip(axes, detectors):
        sub = summary[summary.detector == detector]
        agg = sub.groupby("restorer")[["auroc_clean", "auroc_degraded", "auroc_method"]].mean()
        agg = agg.reindex([r for r in RESTORER_ORDER if r in agg.index])
        x = np.arange(len(agg))
        w = 0.25
        ax.bar(x - w, agg.auroc_clean, width=w, label="clean", color="#999999")
        ax.bar(x, agg.auroc_degraded, width=w, label="degraded", color="#D55E00")
        ax.bar(x + w, agg.auroc_method, width=w, label="restored", color="#009E73")
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{detector}"); ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("image-level AUROC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Clean vs. degraded vs. restored AUROC, pooled across Step 3 scope")
    fig.tight_layout()
    p = out / "step3_auroc_by_restorer.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- companion: scratch DRR and residual correlation by restorer ---
    drr = pd.read_csv(results_dir() / "step3_drr_companion_by_kind.csv")
    scratch = drr[drr.anomaly_kind == "scratch"].set_index("restorer").reindex(RESTORER_ORDER)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    x = np.arange(len(scratch))
    w = 0.35
    ax1.bar(x - w / 2, scratch.relative_drr, width=w, color="#56B4E9", label="relative DRR")
    ax1.axhline(1.0, ls="--", c="green", lw=1.0, alpha=0.7)
    ax1.set_ylabel("relative DRR (1.0 = identity)")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, scratch.residual_corr_mean, width=w, color="#D55E00", label="residual correlation")
    ax2.set_ylabel("mean residual correlation")
    ax1.set_xticks(x); ax1.set_xticklabels(scratch.index, rotation=20, ha="right", fontsize=8)
    ax1.set_title("Scratch defects: relative DRR and residual correlation by restorer\n"
                 "(Step 3 scope: 3 categories x defocus/motion/noise/mixed x severities 2-4)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    p = out / "step3_scratch_drr_corr.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    return paths


def main() -> int:
    paths = make_figures()
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
