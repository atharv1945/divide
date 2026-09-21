"""Harness-correctness gate: does PatchCore reproduce its published MVTec AD
AUROC at the PUBLISHED configuration (full train set, full test set, 256
resolution, default anomalib backbone/coreset)?

Why this exists: eval_grid.py's Step 3 run used n_train=16, n_test=16 per
category - a deliberate speed tradeoff for the restoration-comparison grid,
NOT the configuration anyone would expect to reproduce the ~0.99 published
PatchCore number. That grid reported clean-image AUROC of 0.781 (PatchCore)
and 0.867 (PaDiM), ~20 points below published PatchCore - and this harness
was never checked against the published number at full scale before now.
If this script reproduces roughly the published range, the harness is sound
and the grid's low numbers are attributable to the reduced train/test size
(especially punishing for PatchCore, whose memory bank quality depends on
having enough training patches to subsample a representative coreset from -
16 images is not that). If it does NOT reproduce, the grid results cannot be
trusted until the harness bug is found.

Fits PatchCore once per category on the FULL training split, scores every
image in the FULL test split (good and bad), no restoration, no
degradation - clean images only. Uses the exact same DetectorHarness code
path eval_grid.py uses, so this is a config check, not a different code path.

    python -m src.experiments.patchcore_repro_check
"""
from __future__ import annotations

import json
import time

from src.data.mvtec import load_split, load_train_normals
from src.detect.harness import DetectorHarness
from src.metrics.core import auroc
from src.utils.paths import results_dir

CATEGORIES = ["bottle", "carpet", "screw"]


def run(categories: list[str] = CATEGORIES, size: int = 256) -> dict:
    out = {}
    for category in categories:
        t0 = time.time()
        train_normals = load_train_normals(category, size=size, limit=None, smoke=False)
        test_samples = load_split(category, "test", size=size, limit=None, smoke=False)

        harness = DetectorHarness("patchcore", image_size=size, seed=42)
        harness.fit(train_normals)
        try:
            scores = [harness.score(s.image).score for s in test_samples]
            labels = [s.label for s in test_samples]
        finally:
            harness.close()

        a = auroc(scores, labels)
        el = time.time() - t0
        out[category] = dict(
            auroc=a, n_train=len(train_normals), n_test=len(test_samples),
            n_good=sum(1 for l in labels if l == 0),
            n_bad=sum(1 for l in labels if l == 1),
            elapsed_s=round(el, 1),
        )
        print(f"{category:8s} auroc={a:.4f}  n_train={len(train_normals)} "
              f"n_test={len(test_samples)}  ({el:.0f}s)", flush=True)
    return out


def main() -> int:
    result = run()
    mean_auroc = sum(v["auroc"] for v in result.values()) / len(result)
    result["_mean_auroc"] = mean_auroc
    out_path = results_dir() / "patchcore_repro_check.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nmean AUROC across categories: {mean_auroc:.4f}")
    print(f"result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
