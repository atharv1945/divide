"""THE GO / NO-GO EXPERIMENT.

Measures how much of a defect survives when a standard restorer is applied to
a degraded inspection image.

Method: take a clean normal image, paste a synthetic defect with a known mask,
then degrade BOTH the defect version and the defect-free version with the same
parameters and the same noise seed. Restore both. The Defect Retention Ratio
compares the surviving residual with the original one.

Read the result as follows:

    DRR < 0.5   restorers erase defects. The DIVIDE thesis holds; build it.
    DRR > 0.8   restorers preserve defects. The premise is wrong; pivot now.
    in between  look at the per-restorer and per-anomaly-kind breakdown before
                deciding; scratches are the case that matters most.

Run this before writing any more of the method. It costs two days, and the
alternative is discovering the answer in week eight.

    python -m src.experiments.drr_study --smoke          # no dataset needed
    python -m src.experiments.drr_study --categories carpet bottle screw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.mvtec import DEFAULT_CATEGORIES, load_train_normals, dataset_available
from src.degrade.anomaly import ANOMALY_KINDS, TextureBank, paste_anomaly
from src.degrade.simulator import FAMILIES, SEVERITIES, degrade_pair
from src.dbde.estimator import estimate
from src.experiments.eval_grid import CsvAppender
from src.metrics.core import (
    defect_retention_ratio, defect_residual_correlation, anomaly_contrast_gain,
    degradation_removal_ratio, psnr, ssim,
)
from src.models.restorers import get_restorer, available_restorers
from src.utils.paths import dtd_root, figures_dir, results_dir

ROW_FIELDS = [
    "category", "image", "anomaly_kind", "area_frac", "family", "severity",
    "restorer", "tier", "drr", "residual_corr", "acg", "dremr",
    "psnr_normal", "psnr_full", "ssim_full", "psnr_degraded",
]


def _cell_key(row: dict) -> tuple:
    return (row["category"], row["image"], row["family"], row["severity"], row["restorer"])


def load_completed_cells(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["category", "image", "family", "severity", "restorer"])
    return set(map(tuple, df.drop_duplicates().values.tolist()))


class _Logger:
    """Prints to stdout by default; to a file (append mode) if log_path is
    given, so a long CPU run's progress doesn't get lost or interleaved when
    launched in the background - "logging to file not stdout" per spec."""

    def __init__(self, log_path: Path | None):
        self._fh = open(log_path, "a") if log_path is not None else None

    def __call__(self, msg: str) -> None:
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def run(categories: list[str], restorers: list[str], families: list[str],
        severities: list[int], kinds: list[str], n_images: int,
        size: int, smoke: bool, seed: int, tag: str,
        log_path: Path | None = None) -> Path:
    """Resumable and checkpointed per (category, image, family, severity,
    restorer) cell: results are appended to results/<tag>.csv and flushed
    after every cell, and any cell already present on disk is skipped on
    restart - so a crash mid-run loses at most one in-progress cell, and
    re-running the identical command after a full completion is a no-op
    (mirrors eval_grid.py's CsvAppender/load_completed_cells pattern)."""
    log = _Logger(log_path)
    csv_path = results_dir() / f"{tag}.csv"
    completed = load_completed_cells(csv_path)
    writer = CsvAppender(csv_path, ROW_FIELDS)

    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    if not bank.available:
        log(f"[warn] DTD not found - using procedural texture fallback. "
           f"Expected at {dtd_root()}")

    loaded = {name: get_restorer(name) for name in restorers}
    t0 = time.time()

    total = (len(categories) * n_images * len(families) * len(severities)
             * len(restorers))
    done = 0

    try:
        for cat in categories:
            normals = load_train_normals(cat, size=size, limit=n_images, smoke=smoke)
            if not normals:
                log(f"[warn] no training normals for {cat}")
                continue

            for i, clean0 in enumerate(normals):
                rng = np.random.default_rng(seed + i)
                kind = kinds[i % len(kinds)]
                clean_a, mask, spec = paste_anomaly(clean0, rng, bank=bank, kind=kind)
                if mask.sum() == 0:
                    done += len(families) * len(severities) * len(restorers)
                    continue

                for family in families:
                    for sev in severities:
                        cell_names = [name for name in list(loaded)
                                     if (cat, i, family, sev, name) not in completed]
                        if not cell_names:
                            done += len(restorers)
                            continue

                        y_a, y_0, params = degrade_pair(
                            clean_a, clean0, family, sev, seed=seed + 1000 * i + sev)

                        for name in cell_names:
                            R = loaded[name]
                            done += 1
                            try:
                                r_a = R(y_a)
                                r_0 = R(y_0)
                            except RuntimeError as e:
                                log(f"\n[skip] {name}: {e}")
                                loaded.pop(name, None)
                                break

                            drr = defect_retention_ratio(r_a, r_0, clean_a, clean0, mask)
                            residual_corr = defect_residual_correlation(r_a, r_0, clean_a, clean0, mask)
                            acg = anomaly_contrast_gain(r_a, y_a, mask)
                            dremr = degradation_removal_ratio(r_a, y_a, clean_a, mask)

                            row = dict(
                                category=cat, image=i, anomaly_kind=kind,
                                area_frac=float(mask.mean()),
                                family=family, severity=sev, restorer=name,
                                tier=R.tier,
                                drr=drr, residual_corr=residual_corr, acg=acg, dremr=dremr,
                                psnr_normal=psnr(r_a, clean_a, ~mask.astype(bool)),
                                psnr_full=psnr(r_a, clean_a),
                                ssim_full=ssim(r_a, clean_a),
                                psnr_degraded=psnr(y_a, clean_a),
                            )
                            writer.write_rows([row])
                            completed.add(_cell_key(row))

                        if done % 20 == 0 or done == total:
                            el = time.time() - t0
                            log(f"  {done}/{total} cells  ({el:.0f}s)")
    finally:
        writer.close()
        log.close()

    return csv_path


_EPS = 1e-8


def add_relative_drr(df: pd.DataFrame) -> pd.DataFrame:
    """Joins the matching identity-restorer's raw drr onto each row as
    drr_identity, keyed on (image, family, severity) - the raw ingredient
    every ratio-of-means aggregation below needs. Split out of run() so it
    applies uniformly whether the CSV was just written or read back from a
    previous/resumed run.

    Deliberately does NOT compute a per-row relative-DRR ratio column
    anymore (an earlier version did, called it drr_rel, and every
    aggregate that averaged it was silently computing mean-of-ratios - see
    relative_drr's docstring in metrics/core.py for why that's wrong and
    the concrete number it changed on this project's own data). Every
    consumer below (summarise, verdict, make_figures) computes ratio-of-
    means itself from drr/drr_identity, at whatever grouping it needs -
    there is no single per-row column that would be correct for all of
    them.

    ROW_FIELDS (what run() writes) deliberately excludes drr_identity -
    it's derived, not raw per-cell data - but a CSV from before this
    function existed, or a second call on an already-processed frame, can
    still carry stale drr_identity/drr_rel columns; drop first so this is
    idempotent rather than raising a pandas join-collision on them."""
    if df.empty:
        return df
    df = df.drop(columns=["drr_identity", "drr_rel"], errors="ignore")
    key = ["category", "image", "family", "severity"]
    base = (df[df.restorer == "identity"]
            .set_index(key).drr.rename("drr_identity"))
    if not base.empty:
        df = df.join(base, on=key)
    else:
        df["drr_identity"] = np.nan
    return df


def _ratio_of_means(df: pd.DataFrame) -> float:
    """Relative DRR for a GROUP of rows carrying drr/drr_identity columns
    (see add_relative_drr) - mean(drr)/mean(drr_identity) over the group,
    NOT mean(drr/drr_identity) per row. This project's one aggregation
    rule for relative DRR - see relative_drr's docstring in
    metrics/core.py."""
    if df.empty:
        return float("nan")
    den = float(df["drr_identity"].mean())
    if not np.isfinite(den) or abs(den) <= _EPS:
        return float("nan")
    return float(df["drr"].mean() / den)


def _ratio_of_means_by(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """_ratio_of_means, grouped by `cols` - e.g. ["anomaly_kind"] or
    ["severity"]. Returns a Series indexed by the group key(s) (a
    MultiIndex if len(cols) > 1, e.g. for a restorer x kind pivot via
    .unstack()). Normalises pandas' single-column groupby key (a length-1
    tuple on some pandas versions, a bare scalar on others) to a bare
    scalar either way, so callers get a consistent, plottable Series
    regardless of version."""
    out = {}
    for key, g in df.groupby(cols):
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        out[key] = _ratio_of_means(g)
    return pd.Series(out)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per restorer. drr_rel_mean is RATIO-OF-MEANS (see
    _ratio_of_means) - the median/p10-of-per-example-ratios columns an
    earlier version of this function reported are gone, not silently
    changed: there is no ratio-of-means analogue of a per-example
    percentile, and keeping mean-of-ratios for just those two columns
    while fixing the mean would reintroduce the exact inconsistency this
    was rewritten to remove."""
    rows = []
    for restorer, g in df.groupby("restorer"):
        rows.append(dict(
            restorer=restorer,
            drr_mean=float(g.drr.mean()),
            drr_rel_mean=_ratio_of_means(g),
            residual_corr_mean=(float(g.residual_corr.mean())
                                if "residual_corr" in g and g.residual_corr.notna().any()
                                else float("nan")),
            acg_mean=float(g.acg.mean()),
            dremr_mean=float(g.dremr.mean()),
            psnr_normal=float(g.psnr_normal.mean()),
            n=len(g),
        ))
    return pd.DataFrame(rows).set_index("restorer").sort_values("drr_rel_mean")


def verdict(df: pd.DataFrame) -> tuple[str, str]:
    non_id = df[df.restorer != "identity"]
    if non_id.empty:
        return "INCONCLUSIVE", "no non-identity restorers ran"
    # Judge on RELATIVE DRR (ratio-of-means): the share of defect signal
    # the restorer removed, excluding what the degradation had already
    # destroyed. Falls back to raw drr's own mean only if no identity
    # baseline was present in this data at all (e.g. a restorer list run
    # without "identity" in it).
    m = _ratio_of_means(non_id)
    if not np.isfinite(m):
        m = float(non_id.drr.mean())
    scratch = non_id[non_id.anomaly_kind == "scratch"]
    ms = _ratio_of_means(scratch) if not scratch.empty else float("nan")

    if m < 0.5:
        v = "GO"
        why = (f"mean DRR {m:.3f} < 0.5 - restorers substantially erase defect "
               f"signal. The central claim holds; build DIVIDE.")
    elif m > 0.8:
        v = "NO-GO"
        why = (f"mean DRR {m:.3f} > 0.8 - restorers largely preserve defects. "
               f"The premise does not hold on this setup. Before pivoting, check "
               f"harder degradations and the scratch-only figure below.")
    else:
        v = "MARGINAL"
        why = (f"mean DRR {m:.3f} is in the ambiguous band. Inspect the "
               f"per-restorer and per-kind tables before committing.")
    if np.isfinite(ms):
        why += f"  Scratch-only mean DRR: {ms:.3f} (the case that matters most)."
    return v, why


def make_figures(df: pd.DataFrame, out: Path, tag: str | None = None) -> list[Path]:
    """Writes drr_frontier/drr_vs_severity/drr_by_kind, filenames suffixed
    with `_{tag}` when given. Every call used to write the exact same
    three filenames regardless of which study produced them - the last
    study run (including scripts/smoke.sh's synthetic ones) silently
    overwrote whatever real-data figures a previous run had committed,
    a recurring problem documented in several commits this session.
    Passing a tag makes each study's figures a distinct, non-clobbering
    file; omit it only for throwaway/manual invocations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = f"_{tag}" if tag else ""
    paths = []
    d = df[df.restorer != "identity"]
    if d.empty:
        return paths

    # --- the money figure: preservation-fidelity frontier ---
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    for name, sub in d.groupby("restorer"):
        ax.scatter(sub.psnr_normal.mean(), _ratio_of_means(sub), s=90, label=name,
                   edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(1.0, ls="--", c="green", lw=1.2, alpha=0.7)
    ax.text(ax.get_xlim()[0], 1.005, " perfect defect preservation",
            fontsize=8, color="green", va="bottom")
    ax.axhline(0.5, ls=":", c="red", lw=1.2, alpha=0.7)
    ax.set_xlabel("PSNR on normal regions (dB)  -  degradation removal")
    ax.set_ylabel("Relative DRR  -  defect preservation vs. no-op")
    ax.set_title("Preservation-Fidelity Frontier")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = out / f"drr_frontier{suffix}.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- DRR vs severity ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for name, sub in d.groupby("restorer"):
        s = _ratio_of_means_by(sub, ["severity"]).sort_index()
        ax.plot(s.index, s.values, marker="o", label=name)
    ax.axhline(0.5, ls=":", c="red", lw=1.0)
    ax.set_xlabel("degradation severity"); ax.set_ylabel("mean relative DRR")
    ax.set_title("Defect retention degrades with severity")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / f"drr_vs_severity{suffix}.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- DRR by anomaly kind ---
    if d.anomaly_kind.nunique() > 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        piv = _ratio_of_means_by(d, ["restorer", "anomaly_kind"]).unstack()
        piv.plot.bar(ax=ax, rot=20)
        ax.axhline(0.5, ls=":", c="red", lw=1.0)
        ax.set_ylabel("mean DRR"); ax.set_title("Defect retention by anomaly type")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        p = out / f"drr_by_kind{suffix}.png"
        fig.savefig(p); plt.close(fig); paths.append(p)

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--restorers", nargs="*", default=[
        "identity", "gaussian", "bilateral", "nlm", "clahe",
        "msrcr", "wiener", "classical_pipeline"])
    ap.add_argument("--families", nargs="*", default=["defocus", "noise", "mixed"])
    ap.add_argument("--severities", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--kinds", nargs="*", default=ANOMALY_KINDS)
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="run on synthetic stand-in data, no dataset needed")
    ap.add_argument("--tag", default="drr_study")
    ap.add_argument("--log-file", default=None,
                    help="progress lines go here instead of stdout (results/<tag>.csv "
                         "and the summary/verdict prints at the end are unaffected)")
    args = ap.parse_args()

    if not args.smoke and not dataset_available():
        print("MVTec AD not found. Re-run with --smoke to exercise the pipeline, "
              "or set DIVIDE_DATA_ROOT.", file=sys.stderr)
        return 2

    if args.smoke:
        args.n_images = min(args.n_images, 6)
        args.size = min(args.size, 128)

    print(f"restorers : {args.restorers}")
    print(f"categories: {args.categories}")
    print(f"families  : {args.families}  severities: {args.severities}")
    if args.log_file:
        print(f"progress  : {args.log_file}")
    print()

    log_path = Path(args.log_file) if args.log_file else None
    csv = run(args.categories, args.restorers, args.families, args.severities,
             args.kinds, args.n_images, args.size, args.smoke, args.seed,
             args.tag, log_path=log_path)

    df = add_relative_drr(pd.read_csv(csv))
    if df.empty:
        print("no results produced", file=sys.stderr)
        return 1

    res, figs = results_dir(), figures_dir()

    summary = summarise(df)
    summary.to_csv(res / f"{args.tag}_summary.csv")

    print("\n" + "=" * 74)
    print("DEFECT RETENTION BY RESTORER")
    print("drr_rel: 1.0 = restorer preserved the defect, 0.0 = erased it")
    print("(ratio-of-means: mean(drr)/mean(drr_identity) for the group - see")
    print(" relative_drr's docstring in metrics/core.py. raw drr also counts")
    print(" attenuation caused by the degradation itself.)")
    print("=" * 74)
    print(summary.round(3).to_string())

    if df.anomaly_kind.nunique() > 1:
        print("\nBY ANOMALY TYPE")
        print(_ratio_of_means_by(df[df.restorer != "identity"],
                                 ["restorer", "anomaly_kind"]).unstack().round(3).to_string())

    v, why = verdict(df)
    print("\n" + "=" * 74)
    print(f"VERDICT: {v}")
    print("=" * 74)
    print(why)

    paths = make_figures(df, figs, tag=args.tag)
    print(f"\nrows      : {len(df)}")
    print(f"csv       : {csv}")
    for p in paths:
        print(f"figure    : {p}")

    non_id = df[df.restorer != "identity"]
    (res / f"{args.tag}_verdict.json").write_text(json.dumps(
        dict(verdict=v, reason=why, n_rows=len(df),
             mean_drr=float(non_id.drr.mean()),
             mean_drr_rel=_ratio_of_means(non_id)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
