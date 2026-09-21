"""Bootstrap 95% CIs on gap_closed for every (restorer, detector) pair in
Step 3's grid (results/step3_grid.csv), resampling test images.

Why: gap_closed_agg (eval_grid's ratio-of-means fix) gave point estimates
showing PaDiM negative / PatchCore positive for every restorer - a real,
interesting split IF it survives sampling noise. With only n_test=16 images
per category, reused across every family/severity cell for that category,
the grid could easily be underpowered to distinguish that split from zero.
This resamples which of the 16 physical test images are included (with
replacement) - the SAME resampled index set applied across every
family/severity cell for a category, since those cells reuse the identical
16 physical images (see eval_grid.py's test_samples construction) - and
recomputes gap_closed_agg (mean(method-degraded)/mean(clean-degraded),
pooled over all cells for that restorer/detector) on each resample. This is
the correct resampling unit: resampling individual (image, cell) rows
independently would treat the same physical image's repeated appearances
across 12 family/severity cells as independent draws, understating the true
sampling variance.

    python -m src.experiments.bootstrap_grid_gapclosed
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.metrics.core import auroc
from src.utils.paths import results_dir

N_BOOT = 2000
SEED = 42


def _cell_arrays(df: pd.DataFrame) -> dict:
    """Indexes every (category, detector, family, severity, restorer) cell's
    per-image score/label arrays by image_id 0..n_test-1, so a bootstrap
    draw of image indices can be applied directly without re-filtering the
    dataframe on every iteration."""
    cells = {}
    for (cat, det, fam, sev), g in df.groupby(["category", "detector", "family", "severity"]):
        g0 = g[g.restorer == g.restorer.iloc[0]].sort_values("image_id")
        n = g0.image_id.nunique()
        labels = g0.set_index("image_id").label.reindex(range(n)).to_numpy()
        clean = g0.set_index("image_id").score_clean.reindex(range(n)).to_numpy()
        degraded = g0.set_index("image_id").score_degraded.reindex(range(n)).to_numpy()
        methods = {}
        for restorer, gr in g.groupby("restorer"):
            gr = gr.sort_values("image_id").set_index("image_id")
            methods[restorer] = gr.score_method.reindex(range(n)).to_numpy()
        cells[(cat, det, fam, sev)] = dict(n=n, labels=labels, clean=clean,
                                           degraded=degraded, methods=methods)
    return cells


def bootstrap(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    cells = _cell_arrays(df)
    categories = sorted({k[0] for k in cells})
    detectors = sorted(df.detector.unique())
    restorers = sorted(df.restorer.unique())
    n_test = {c: cells[k]["n"] for k in cells for c in [k[0]] if k[0] == c}
    # every cell for a given category shares the same n_test (see module docstring)
    n_by_cat = {cat: next(v["n"] for k, v in cells.items() if k[0] == cat) for cat in categories}

    rng = np.random.default_rng(seed)
    point = _gap_closed_point(cells, detectors, restorers)

    draws = {(restorer, detector): np.empty(n_boot) for restorer in restorers for detector in detectors}
    for b in range(n_boot):
        idx_by_cat = {cat: rng.integers(0, n_by_cat[cat], size=n_by_cat[cat]) for cat in categories}
        for detector in detectors:
            for restorer in restorers:
                num, den = [], []
                for (cat, det, fam, sev), c in cells.items():
                    if det != detector:
                        continue
                    idx = idx_by_cat[cat]
                    y = c["labels"][idx]
                    ac = auroc(c["clean"][idx], y)
                    ad = auroc(c["degraded"][idx], y)
                    am = auroc(c["methods"][restorer][idx], y)
                    if np.isfinite(ac) and np.isfinite(ad) and np.isfinite(am):
                        num.append(am - ad)
                        den.append(ac - ad)
                gc = (np.mean(num) / np.mean(den)) if den and abs(np.mean(den)) > 1e-8 else np.nan
                draws[(restorer, detector)][b] = gc

    rows = []
    for (restorer, detector), d in draws.items():
        d = d[np.isfinite(d)]
        lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5])) if len(d) else (float("nan"), float("nan"))
        rows.append(dict(
            restorer=restorer, detector=detector,
            gap_closed_point=point[(restorer, detector)],
            ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0 or hi < 0),
            n_boot_valid=len(d), n_boot=n_boot,
        ))
    return pd.DataFrame(rows)


def _gap_closed_point(cells: dict, detectors: list[str], restorers: list[str]) -> dict:
    out = {}
    for detector in detectors:
        for restorer in restorers:
            num, den = [], []
            for (cat, det, fam, sev), c in cells.items():
                if det != detector:
                    continue
                y = c["labels"]
                ac = auroc(c["clean"], y)
                ad = auroc(c["degraded"], y)
                am = auroc(c["methods"][restorer], y)
                if np.isfinite(ac) and np.isfinite(ad) and np.isfinite(am):
                    num.append(am - ad)
                    den.append(ac - ad)
            out[(restorer, detector)] = (np.mean(num) / np.mean(den)) if den and abs(np.mean(den)) > 1e-8 else float("nan")
    return out


def main() -> int:
    df = pd.read_csv(results_dir() / "step3_grid.csv")
    result = bootstrap(df)
    result = result.sort_values(["detector", "restorer"])
    print(result.round(3).to_string(index=False))
    out_path = results_dir() / "step3_grid_gapclosed_bootstrap.csv"
    result.to_csv(out_path, index=False)
    print(f"\nresult: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
