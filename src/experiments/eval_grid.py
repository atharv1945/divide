"""Frozen-detector evaluation grid.

Restorers x detectors x degradation families x severities x categories.
For each category, each frozen detector is fit ONCE on clean training
normals and reused across every restorer/family/severity cell - the whole
point of the frozen-detector harness. Each test image (good or defective)
is scored three ways: clean (undegraded), degraded (no restoration), and
restored (through the cell's restorer). gap_closed() then says what
fraction of the clean-to-degraded AUROC drop each restorer recovered.

Resumable: the CSV is keyed by (restorer, detector, category, family,
severity, anomaly_kind) per-image, and a "cell" is one (restorer, detector,
category, family, severity) combination - the actual grid dimensions. On
restart, any cell already present in the CSV is skipped; rows are appended
and flushed to disk after every cell, so a crash loses at most one
in-progress cell, not the whole run.

    python -m src.experiments.eval_grid --smoke                   # CPU, no data/weights
    python -m src.experiments.eval_grid --categories carpet bottle screw \
        --detectors padim --restorers identity gaussian nlm classical_pipeline
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.mvtec import DEFAULT_CATEGORIES, load_split, load_train_normals, dataset_available
from src.degrade.simulator import FAMILIES, SEVERITIES, degrade
from src.detect.harness import AVAILABLE as AVAILABLE_DETECTORS, DetectorHarness
from src.metrics.core import auroc, gap_closed
from src.models.restorers import get_restorer
from src.utils.paths import results_dir

ROW_FIELDS = [
    "restorer", "detector", "category", "family", "severity",
    "anomaly_kind", "image_id", "label",
    "score_clean", "score_degraded", "score_method",
]

# Companion, cell-level (not per-image) CSV. Pixel-level AUROC is computed
# from anomaly_map, already returned by DetectorHarness.score() at no extra
# inference cost - but a (H,W) float32 map per image per condition is too
# large to persist per-image-row (256x256x3 conditions x every image), so
# only the pooled-per-cell AUROC survives to disk; the maps themselves stay
# in memory for exactly as long as one cell takes to score.
PIXEL_ROW_FIELDS = [
    "restorer", "detector", "category", "family", "severity",
    "n_pixel_images", "pix_auroc_clean", "pix_auroc_degraded",
    "pix_auroc_method", "pix_gap_closed",
]


def _balanced_test_subset(samples: list, n_test: int) -> list:
    """load_split()'s `limit` truncates the concatenated (good, then bad)
    list, so a small limit can silently take ALL-good or ALL-bad and leave
    AUROC undefined for every cell. Take up to n_test//2 of each label
    instead, so both classes are present whenever the category has any."""
    good = [s for s in samples if s.label == 0][: max(n_test // 2, 1)]
    bad = [s for s in samples if s.label == 1][: max(n_test - len(good), 1)]
    return good + bad


def _cell_key(row: dict) -> tuple:
    return (row["restorer"], row["detector"], row["category"], row["family"], row["severity"])


def load_completed_cells(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["restorer", "detector", "category", "family", "severity"])
    return set(map(tuple, df.drop_duplicates().values.tolist()))


def _pixel_ground_truth(samples: list, size: int) -> dict[int, np.ndarray]:
    """Per-sample (H,W) binary ground truth for pixel-level AUROC. Good
    images (label 0) have no mask on disk by construction - they're
    genuinely defect-free, so the correct ground truth is all-zero, not
    "missing". Bad images (label 1) without a mask (can happen in smoke
    mode, where masks aren't synthesised) are dropped from the pixel-level
    pool entirely rather than guessed at - only images with a real mask
    contribute to pixel AUROC."""
    out = {}
    for i, s in enumerate(samples):
        if s.mask is not None:
            out[i] = s.mask
        elif s.label == 0:
            out[i] = np.zeros((size, size), dtype=np.uint8)
    return out


def _reconcile_partial_cells(csv_path: Path, pixel_csv_path: Path) -> set[tuple]:
    """A cell writes its image rows, then its one pixel row - not
    atomically. A crash between the two leaves image rows for a cell the
    pixel CSV never recorded as done; on resume that cell would otherwise
    look complete from the image CSV alone (silently under-counting) or
    get re-run and DUPLICATE its image rows (silently over-counting) once
    a pixel row also existed. Fix: pixel-completion is authoritative (it's
    written last); drop any image rows for cells NOT in it so the normal
    per-cell loop regenerates them cleanly. Returns the reconciled
    completed-cells set."""
    pixel_completed = load_completed_cells(pixel_csv_path)
    if not csv_path.exists():
        return pixel_completed
    df = pd.read_csv(csv_path)
    if df.empty:
        return pixel_completed
    keys = list(zip(df.restorer, df.detector, df.category, df.family, df.severity))
    keep = [k in pixel_completed for k in keys]
    if not all(keep):
        df[keep].to_csv(csv_path, index=False)
    return pixel_completed


def _pixel_auroc(maps: dict[int, np.ndarray], masks: dict[int, np.ndarray]) -> tuple[float, int]:
    """Pools per-pixel scores/labels across every image with a usable mask
    and computes one AUROC over the concatenation - the standard pixel-
    level AUROC definition, not an average of per-image AUROCs."""
    idxs = sorted(masks)
    if not idxs:
        return float("nan"), 0
    scores = np.concatenate([np.asarray(maps[i]).ravel() for i in idxs])
    labels = np.concatenate([masks[i].ravel() for i in idxs])
    return auroc(scores, labels), len(idxs)


class CsvAppender:
    """Append-and-flush CSV writer so a crash mid-run keeps every completed row."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        is_new = not path.exists()
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write_rows(self, rows: list[dict]) -> None:
        for r in rows:
            self._writer.writerow(r)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run(categories: list[str], detectors: list[str], restorers: list[str],
        families: list[str], severities: list[int], n_train: int, n_test: int,
        size: int, smoke: bool, seed: int, tag: str) -> tuple[Path, Path]:
    csv_path = results_dir() / f"{tag}.csv"
    pixel_csv_path = results_dir() / f"{tag}_pixel.csv"
    completed = _reconcile_partial_cells(csv_path, pixel_csv_path)
    writer = CsvAppender(csv_path, ROW_FIELDS)
    pixel_writer = CsvAppender(pixel_csv_path, PIXEL_ROW_FIELDS)

    total_cells = len(categories) * len(detectors) * len(families) * len(severities) * len(restorers)
    done_cells = 0
    t0 = time.time()

    try:
        for category in categories:
            train_normals = load_train_normals(category, size=size, limit=n_train, smoke=smoke)
            # Load more than n_test and then subsample for a good/bad mix,
            # rather than the full split (could be hundreds of images on
            # real MVTec) - load_split's own directory-order truncation
            # otherwise risks an all-one-class slice and undefined AUROC.
            candidates = load_split(category, "test", size=size,
                                    limit=max(n_test * 20, 200), smoke=smoke)
            test_samples = _balanced_test_subset(candidates, n_test)
            if not train_normals or not test_samples:
                print(f"[warn] no data for category {category!r}, skipping", file=sys.stderr)
                continue
            pixel_masks = _pixel_ground_truth(test_samples, size)

            for detector_name in detectors:
                cells_for_detector = [
                    (family, sev, restorer_name)
                    for family in families for sev in severities for restorer_name in restorers
                    if (restorer_name, detector_name, category, family, sev) not in completed
                ]
                if not cells_for_detector:
                    done_cells += len(families) * len(severities) * len(restorers)
                    continue

                harness = DetectorHarness(detector_name, image_size=size, seed=seed)
                harness.fit(train_normals)
                try:
                    clean_results = {i: harness.score(s.image) for i, s in enumerate(test_samples)}
                    clean_scores = {i: r.score for i, r in clean_results.items()}
                    clean_maps = {i: r.anomaly_map for i, r in clean_results.items()}

                    for family in families:
                        for sev in severities:
                            degraded, degraded_scores, degraded_maps = {}, {}, {}
                            for i, s in enumerate(test_samples):
                                y, _ = degrade(s.image, family, sev, seed=seed + i)
                                degraded[i] = y
                                r = harness.score(y)
                                degraded_scores[i] = r.score
                                degraded_maps[i] = r.anomaly_map
                            auroc_clean_pix, _ = _pixel_auroc(clean_maps, pixel_masks)
                            auroc_degraded_pix, n_pix = _pixel_auroc(degraded_maps, pixel_masks)

                            for restorer_name in restorers:
                                key = (restorer_name, detector_name, category, family, sev)
                                done_cells += 1
                                if key in completed:
                                    continue
                                try:
                                    restorer = get_restorer(restorer_name)
                                except KeyError as e:
                                    print(f"[skip] {e}", file=sys.stderr)
                                    continue

                                rows = []
                                method_maps = {}
                                cell_failed = False
                                for i, s in enumerate(test_samples):
                                    try:
                                        x_tilde = restorer(degraded[i])
                                        method_result = harness.score(x_tilde)
                                    except RuntimeError as e:
                                        print(f"\n[skip] {restorer_name}: {e}", file=sys.stderr)
                                        cell_failed = True
                                        break
                                    method_maps[i] = method_result.anomaly_map
                                    rows.append(dict(
                                        restorer=restorer_name, detector=detector_name,
                                        category=category, family=family, severity=sev,
                                        anomaly_kind=s.defect, image_id=i, label=s.label,
                                        score_clean=clean_scores[i],
                                        score_degraded=degraded_scores[i],
                                        score_method=method_result.score,
                                    ))
                                if cell_failed:
                                    break  # missing weights etc. - stop this restorer entirely
                                writer.write_rows(rows)

                                auroc_method_pix, _ = _pixel_auroc(method_maps, pixel_masks)
                                pixel_writer.write_rows([dict(
                                    restorer=restorer_name, detector=detector_name,
                                    category=category, family=family, severity=sev,
                                    n_pixel_images=n_pix,
                                    pix_auroc_clean=auroc_clean_pix,
                                    pix_auroc_degraded=auroc_degraded_pix,
                                    pix_auroc_method=auroc_method_pix,
                                    pix_gap_closed=gap_closed(auroc_clean_pix, auroc_degraded_pix,
                                                              auroc_method_pix),
                                )])
                                completed.add(key)

                                el = time.time() - t0
                                print(f"  cell {done_cells}/{total_cells} "
                                      f"({detector_name}/{category}/{family}/sev{sev}/{restorer_name}) "
                                      f"({el:.0f}s)", flush=True)
                finally:
                    harness.close()
    finally:
        writer.close()
        pixel_writer.close()

    return csv_path, pixel_csv_path


def summarise(csv_path: Path) -> pd.DataFrame:
    """gap_closed() per (restorer, detector, category, family, severity),
    computed by pooling per-image scores within that cell (across
    anomaly_kind, since AUROC needs both normal and defective examples)."""
    df = pd.read_csv(csv_path)
    rows = []
    key_cols = ["detector", "category", "family", "severity"]
    for key, g in df.groupby(key_cols):
        detector, category, family, severity = key
        auroc_clean = auroc(g.score_clean.values, g.label.values)
        auroc_degraded = auroc(g.score_degraded.values, g.label.values)
        for restorer_name, gr in g.groupby("restorer"):
            auroc_method = auroc(gr.score_method.values, gr.label.values)
            rows.append(dict(
                detector=detector, category=category, family=family, severity=severity,
                restorer=restorer_name, n=len(gr),
                auroc_clean=auroc_clean, auroc_degraded=auroc_degraded,
                auroc_method=auroc_method,
                gap_closed=gap_closed(auroc_clean, auroc_degraded, auroc_method),
            ))
    return pd.DataFrame(rows)


def merge_pixel_summary(summary: pd.DataFrame, pixel_csv_path: Path) -> pd.DataFrame:
    """Left-joins the cell-level pixel CSV (already one row per cell, no
    further aggregation needed) onto the image-level summary. A restorer
    that failed (missing weights) has image-level rows but no pixel rows
    for that cell - those pixel_* columns come back NaN, not a crash."""
    if not pixel_csv_path.exists():
        return summary
    pix = pd.read_csv(pixel_csv_path)
    if pix.empty:
        return summary
    key_cols = ["detector", "category", "family", "severity", "restorer"]
    return summary.merge(pix, on=key_cols, how="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--detectors", nargs="*", default=["padim"],
                    help=f"any of {AVAILABLE_DETECTORS}")
    ap.add_argument("--restorers", nargs="*", default=[
        "identity", "gaussian", "nlm", "clahe", "wiener", "classical_pipeline"])
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--severities", nargs="*", type=int, default=SEVERITIES)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="eval_grid")
    args = ap.parse_args()

    if not args.smoke and not dataset_available():
        print("MVTec AD not found. Re-run with --smoke, or set DIVIDE_DATA_ROOT.",
              file=sys.stderr)
        return 2

    if args.smoke:
        args.n_train = min(args.n_train, 6)
        args.n_test = min(args.n_test, 6)
        args.size = min(args.size, 64)  # ReverseDistillation needs a multiple of 32

    print(f"detectors : {args.detectors}")
    print(f"restorers : {args.restorers}")
    print(f"categories: {args.categories}")
    print(f"families  : {args.families}  severities: {args.severities}")
    print()

    csv_path, pixel_csv_path = run(args.categories, args.detectors, args.restorers,
                                   args.families, args.severities, args.n_train,
                                   args.n_test, args.size, args.smoke, args.seed, args.tag)

    summary = summarise(csv_path)
    if summary.empty:
        print("no results produced", file=sys.stderr)
        return 1
    summary = merge_pixel_summary(summary, pixel_csv_path)
    summary_path = results_dir() / f"{args.tag}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 74)
    print("IMAGE-LEVEL GAP CLOSED BY RESTORER (1.0 = fully recovered clean-detector AUROC)")
    print("=" * 74)
    piv = summary.pivot_table(index="restorer", columns="detector",
                              values="gap_closed", aggfunc="mean")
    print(piv.round(3).to_string())

    if "pix_gap_closed" in summary.columns and summary["pix_gap_closed"].notna().any():
        print("\n" + "=" * 74)
        print("PIXEL-LEVEL GAP CLOSED BY RESTORER")
        print("=" * 74)
        piv_pix = summary.pivot_table(index="restorer", columns="detector",
                                      values="pix_gap_closed", aggfunc="mean")
        print(piv_pix.round(3).to_string())

    print(f"\ncsv       : {csv_path}")
    print(f"pixel csv : {pixel_csv_path}")
    print(f"summary   : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
