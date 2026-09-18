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
from src.metrics.core import (
    defect_retention_ratio, anomaly_contrast_gain,
    degradation_removal_ratio, relative_drr, psnr, ssim,
)
from src.models.restorers import get_restorer, available_restorers
from src.utils.paths import dtd_root, figures_dir, results_dir


def run(categories: list[str], restorers: list[str], families: list[str],
        severities: list[int], kinds: list[str], n_images: int,
        size: int, smoke: bool, seed: int) -> pd.DataFrame:

    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    if not bank.available:
        print("[warn] DTD not found - using procedural texture fallback. "
              f"Expected at {dtd_root()}", file=sys.stderr)

    loaded = {name: get_restorer(name) for name in restorers}
    rows: list[dict] = []
    t0 = time.time()

    total = (len(categories) * n_images * len(families) * len(severities)
             * len(restorers))
    done = 0

    for cat in categories:
        normals = load_train_normals(cat, size=size, limit=n_images, smoke=smoke)
        if not normals:
            print(f"[warn] no training normals for {cat}", file=sys.stderr)
            continue

        for i, clean0 in enumerate(normals):
            rng = np.random.default_rng(seed + i)
            kind = kinds[i % len(kinds)]
            clean_a, mask, spec = paste_anomaly(clean0, rng, bank=bank, kind=kind)
            if mask.sum() == 0:
                continue

            for family in families:
                for sev in severities:
                    y_a, y_0, params = degrade_pair(
                        clean_a, clean0, family, sev, seed=seed + 1000 * i + sev)

                    for name, R in loaded.items():
                        done += 1
                        try:
                            r_a = R(y_a)
                            r_0 = R(y_0)
                        except RuntimeError as e:
                            print(f"\n[skip] {name}: {e}", file=sys.stderr)
                            loaded.pop(name, None)
                            break

                        drr = defect_retention_ratio(r_a, r_0, clean_a, clean0, mask)
                        acg = anomaly_contrast_gain(r_a, y_a, mask)
                        dremr = degradation_removal_ratio(r_a, y_a, clean_a, mask)

                        rows.append(dict(
                            category=cat, image=i, anomaly_kind=kind,
                            area_frac=float(mask.mean()),
                            family=family, severity=sev, restorer=name,
                            tier=R.tier,
                            drr=drr, acg=acg, dremr=dremr,
                            psnr_normal=psnr(r_a, clean_a, ~mask.astype(bool)),
                            psnr_full=psnr(r_a, clean_a),
                            ssim_full=ssim(r_a, clean_a),
                            psnr_degraded=psnr(y_a, clean_a),
                        ))

                    if done % 200 == 0:
                        el = time.time() - t0
                        print(f"  {done}/{total} cells  ({el:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalise against the identity baseline per (image, family, severity):
    # raw DRR includes attenuation caused by the degradation itself, which no
    # restorer is responsible for. See metrics.relative_drr.
    key = ["category", "image", "family", "severity"]
    base = (df[df.restorer == "identity"]
            .set_index(key).drr.rename("drr_identity"))
    if not base.empty:
        df = df.join(base, on=key)
        df["drr_rel"] = [relative_drr(a, b)
                         for a, b in zip(df.drr, df.drr_identity)]
    else:
        df["drr_identity"] = np.nan
        df["drr_rel"] = np.nan
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("restorer").agg(
        drr_mean=("drr", "mean"),
        drr_rel_mean=("drr_rel", "mean"),
        drr_rel_median=("drr_rel", "median"),
        drr_rel_p10=("drr_rel", lambda s: s.quantile(0.10)),
        acg_mean=("acg", "mean"),
        dremr_mean=("dremr", "mean"),
        psnr_normal=("psnr_normal", "mean"),
        n=("drr", "size"),
    ).sort_values("drr_rel_mean")
    return g


def verdict(df: pd.DataFrame) -> tuple[str, str]:
    non_id = df[df.restorer != "identity"]
    if non_id.empty:
        return "INCONCLUSIVE", "no non-identity restorers ran"
    # Judge on RELATIVE DRR: the share of defect signal the restorer removed,
    # excluding what the degradation had already destroyed.
    col = "drr_rel" if non_id.drr_rel.notna().any() else "drr"
    m = float(non_id[col].mean())
    scratch = non_id[non_id.anomaly_kind == "scratch"]
    ms = float(scratch[col].mean()) if not scratch.empty else float("nan")

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


def make_figures(df: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    d = df[df.restorer != "identity"]
    if d.empty:
        return paths

    # --- the money figure: preservation-fidelity frontier ---
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    for name, sub in d.groupby("restorer"):
        ax.scatter(sub.psnr_normal.mean(), sub.drr_rel.mean(), s=90, label=name,
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
    p = out / "drr_frontier.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- DRR vs severity ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for name, sub in d.groupby("restorer"):
        s = sub.groupby("severity").drr_rel.mean()
        ax.plot(s.index, s.values, marker="o", label=name)
    ax.axhline(0.5, ls=":", c="red", lw=1.0)
    ax.set_xlabel("degradation severity"); ax.set_ylabel("mean relative DRR")
    ax.set_title("Defect retention degrades with severity")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "drr_vs_severity.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # --- DRR by anomaly kind ---
    if d.anomaly_kind.nunique() > 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        piv = d.pivot_table(index="restorer", columns="anomaly_kind",
                            values="drr_rel", aggfunc="mean")
        piv.plot.bar(ax=ax, rot=20)
        ax.axhline(0.5, ls=":", c="red", lw=1.0)
        ax.set_ylabel("mean DRR"); ax.set_title("Defect retention by anomaly type")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        p = out / "drr_by_kind.png"
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
    print()

    df = run(args.categories, args.restorers, args.families, args.severities,
             args.kinds, args.n_images, args.size, args.smoke, args.seed)

    if df.empty:
        print("no results produced", file=sys.stderr)
        return 1

    res, figs = results_dir(), figures_dir()
    csv = res / f"{args.tag}.csv"
    df.to_csv(csv, index=False)

    summary = summarise(df)
    summary.to_csv(res / f"{args.tag}_summary.csv")

    print("\n" + "=" * 74)
    print("DEFECT RETENTION BY RESTORER")
    print("drr_rel: 1.0 = restorer preserved the defect, 0.0 = erased it")
    print("(raw drr also counts attenuation caused by the degradation itself)")
    print("=" * 74)
    print(summary.round(3).to_string())

    if df.anomaly_kind.nunique() > 1:
        print("\nBY ANOMALY TYPE")
        print(df[df.restorer != "identity"]
              .pivot_table(index="restorer", columns="anomaly_kind",
                           values="drr_rel", aggfunc="mean").round(3).to_string())

    v, why = verdict(df)
    print("\n" + "=" * 74)
    print(f"VERDICT: {v}")
    print("=" * 74)
    print(why)

    paths = make_figures(df, figs)
    print(f"\nrows      : {len(df)}")
    print(f"csv       : {csv}")
    for p in paths:
        print(f"figure    : {p}")

    (res / f"{args.tag}_verdict.json").write_text(json.dumps(
        dict(verdict=v, reason=why, n_rows=len(df),
             mean_drr=float(df[df.restorer != 'identity'].drr.mean()),
             mean_drr_rel=float(df[df.restorer != 'identity'].drr_rel.mean())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
