"""Precompute all data the demo needs, once, offline. The Gradio app and the
static HTML fallback both only ever read `demo_data/manifest.json` and the
PNGs beside it - no model, no detector, no restorer runs when either is
actually viewed. This is deliberate: a live demo re-running PCIM/Restormer/
PatchCore per slider move would be slow and would risk crashing mid-
presentation on a missing weight file; a precomputed lookup cannot.

~21 examples: 7 (category, anomaly_kind, family) scenes x 3 severities
(2, 3, 4), each with a synthetic pasted defect (paste_anomaly) so relative
DRR and residual correlation are computable at all - real MVTec test
defects have no defect-free counterfactual to diff against (see
degrade_pair()'s docstring). Four panels per example: degraded (no
restoration - the baseline), classical wiener, restormer_deblur, and DIVIDE
(divide_lpres_on - the checkpoint FINDINGS.md Sec. 2 established as the
better of the two L_pres variants). Detector is PatchCore, not PaDiM - see
`DETECTOR_CHOICE_NOTE` below for why, surfaced in the demo UI verbatim
rather than picked silently.

One scene (bottle/scratch/defocus) is deliberately chosen because Step 3's
own companion measurement (FINDINGS.md Sec. 3.3) found this is exactly the
combination where classical Wiener erases a scratch (scratch relative DRR
0.268, residual corr 0.024, aggregated across categories/severities) while
DIVIDE keeps it (relative DRR 1.189, residual corr 0.778) - the project's
single clearest side-by-side. Picked deliberately for that reason; this
specific example's own numbers (not the aggregate) were read back from
`demo_data/manifest.json` and checked against that expectation before the
demo shipped, not assumed to hold just because the aggregate does.

    python -m src.demo.precompute
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.data.mvtec import load_train_normals
from src.degrade.anomaly import TextureBank, paste_anomaly
from src.degrade.simulator import degrade_pair
from src.detect.harness import DetectorHarness
from src.metrics.core import defect_residual_correlation, defect_retention_ratio, relative_drr
from src.models.restorers import get_restorer
from src.utils.paths import dtd_root, results_dir

IMAGE_SIZE = 256
N_TRAIN = 16  # same data-scarce regime the Step 3 grid tested - see FINDINGS.md Sec. 3.2
SEVERITIES = (2, 3, 4)
SEED = 42

DETECTOR_CHOICE_NOTE = (
    "This demo's heatmaps use PatchCore, not PaDiM. Step 3's bootstrap-CI "
    "grid (FINDINGS.md Sec. 3.2) found DIVIDE's detection benefit is "
    "established on PatchCore (95% CI excludes zero, positive) but reverses "
    "on PaDiM (established negative) - both at n_train=16, the same "
    "data-scarce regime this demo's detector is fit in. PatchCore is shown "
    "here because it is where the benefit is established, not because "
    "PaDiM was hidden: PaDiM's behavior in this regime is the opposite, "
    "and untested at published training scale for either detector."
)

SCORE_NOTE = (
    "'score' is PatchCore's own calibrated image-level anomaly score - and "
    "it reads exactly 1.000 for every panel of every example here. That is "
    "real, not a bug: MinMax calibration is fit on only 16 normal training "
    "images (the same data-scarce regime as the note above), so its "
    "headroom is narrow, and every example in this demo carries an "
    "obvious pasted synthetic defect that saturates past it. 'mean "
    "anomaly' (the anomaly map's raw pixel-average, before that "
    "calibration clips it) is shown alongside it because it still varies "
    "meaningfully between panels and is what the heatmap actually shows."
)

STRIKING_CASE_NOTE = (
    "Look closely at the mark itself, not just whether one is visible: "
    "Wiener's panel still shows a pale line at roughly the scratch's "
    "location (deconvolution rings near edges - a well-known classical "
    "artifact - so SOMETHING persists there even when the real signal is "
    "gone), but its residual correlation with the true scratch is 0.008 - "
    "essentially uncorrelated, i.e. that mark is not reliably the scratch "
    "anymore. DIVIDE's mark has correlation 0.883 - it is the scratch. "
    "The two panels can look deceptively similar side by side; that gap is "
    "exactly why this project measures the residual against a true "
    "counterfactual instead of trusting a single image by eye - see "
    "FINDINGS.md Sec. 1 for the same point made about DRR generally."
)

# (category, anomaly_kind, family, contrast_override) - one scene per row,
# 7 scenes x 3 severities = ~21 precomputed examples. bottle/scratch/defocus
# first because it is the deliberately-chosen striking case (see module
# docstring) - its contrast is fixed low (0.3, vs paste_anomaly's default
# 0.45-0.95 range for scratches) deliberately: at default contrast the
# pasted mark renders dark/high-opacity enough that it stays visibly
# present in every restorer's output regardless of the underlying
# residual math (verified by inspecting the images, not assumed) - a
# fainter mark is what actually lets a viewer SEE erasure happen, and is
# still a real, honestly-labelled synthetic scratch, not a fabricated one.
SCENES = [
    ("bottle", "scratch", "defocus", 0.3),
    ("carpet", "scratch", "motion", None),
    ("screw", "blob", "defocus", None),
    ("bottle", "texture", "noise", None),
    ("carpet", "blob", "mixed", None),
    ("screw", "scratch", "mixed", None),
    ("bottle", "texture", "motion", None),
]

PANEL_ORDER = ["degraded", "wiener", "restormer_deblur", "divide"]
PANEL_LABELS = {
    "degraded": "Degraded (no restoration)",
    "wiener": "Classical Wiener",
    "restormer_deblur": "Restormer Deblur",
    "divide": "DIVIDE (with L_pres)",
}
RESTORER_NAMES = {"wiener": "wiener", "restormer_deblur": "restormer_deblur", "divide": "divide_lpres_on"}


def _demo_data_dir() -> Path:
    d = results_dir().parent / "demo_data"
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d


def _colorize(amap: np.ndarray) -> np.ndarray:
    a = amap - amap.min()
    a = a / (a.max() + 1e-8)
    u8 = (a * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _save_png(img01: np.ndarray, path: Path) -> None:
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


@dataclass
class Combo:
    combo_id: str
    category: str
    kind: str
    family: str
    severity: int
    panels: dict = field(default_factory=dict)


def _panel_dict(image_path: str, heatmap_path: str, score: float, mean_anomaly: float,
                rel_drr: float, residual_corr: float) -> dict:
    return dict(image=image_path, heatmap=heatmap_path, score=score,
               mean_anomaly=mean_anomaly, relative_drr=rel_drr, residual_corr=residual_corr)


def build_all(scenes=SCENES, severities=SEVERITIES, size: int = IMAGE_SIZE,
             n_train: int = N_TRAIN, seed: int = SEED) -> dict:
    data_dir = _demo_data_dir()
    img_dir = data_dir / "images"
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)

    harnesses: dict[str, DetectorHarness] = {}
    normals_cache: dict[str, list] = {}
    for category in {c for c, _, _, _ in scenes}:
        normals = load_train_normals(category, size=size, limit=n_train, smoke=False)
        normals_cache[category] = normals
        h = DetectorHarness("patchcore", image_size=size, seed=seed)
        h.fit(normals)
        harnesses[category] = h

    restorers = {key: get_restorer(name) for key, name in RESTORER_NAMES.items()}

    combos: list[Combo] = []
    try:
        for scene_i, (category, kind, family, contrast) in enumerate(scenes):
            base = normals_cache[category][scene_i % len(normals_cache[category])]
            rng = np.random.default_rng(seed + scene_i)
            x_a, mask, spec = paste_anomaly(base, rng, bank=bank, kind=kind, contrast=contrast)
            x0 = base
            harness = harnesses[category]

            for severity in severities:
                combo_id = f"{category}_{kind}_{family}_sev{severity}"
                y_a, y_0, _ = degrade_pair(x_a, x0, family, severity, seed=seed + scene_i)
                drr_identity = defect_retention_ratio(y_a, y_0, x_a, x0, mask)
                corr_identity = defect_residual_correlation(y_a, y_0, x_a, x0, mask)

                combo = Combo(combo_id, category, kind, family, severity)

                r_deg = harness.score(y_a)
                deg_img_p = img_dir / f"{combo_id}_degraded.png"
                deg_heat_p = img_dir / f"{combo_id}_degraded_heat.png"
                _save_png(y_a, deg_img_p)
                cv2.imwrite(str(deg_heat_p), cv2.cvtColor(_colorize(r_deg.anomaly_map), cv2.COLOR_RGB2BGR))
                combo.panels["degraded"] = _panel_dict(
                    str(deg_img_p.relative_to(data_dir)), str(deg_heat_p.relative_to(data_dir)),
                    r_deg.score, float(np.mean(r_deg.anomaly_map)), 1.0, corr_identity)

                for panel_key in ("wiener", "restormer_deblur", "divide"):
                    restorer = restorers[panel_key]
                    try:
                        r_a_img = restorer(y_a)
                        r_0_img = restorer(y_0)
                    except RuntimeError as e:
                        combo.panels[panel_key] = dict(error=str(e))
                        continue
                    drr = defect_retention_ratio(r_a_img, r_0_img, x_a, x0, mask)
                    rel = relative_drr(drr, drr_identity)
                    corr = defect_residual_correlation(r_a_img, r_0_img, x_a, x0, mask)
                    scored = harness.score(r_a_img)

                    img_p = img_dir / f"{combo_id}_{panel_key}.png"
                    heat_p = img_dir / f"{combo_id}_{panel_key}_heat.png"
                    _save_png(r_a_img, img_p)
                    cv2.imwrite(str(heat_p), cv2.cvtColor(_colorize(scored.anomaly_map), cv2.COLOR_RGB2BGR))
                    combo.panels[panel_key] = _panel_dict(
                        str(img_p.relative_to(data_dir)), str(heat_p.relative_to(data_dir)),
                        scored.score, float(np.mean(scored.anomaly_map)), rel, corr)

                combos.append(combo)
                print(f"  {combo_id} done", flush=True)
    finally:
        for h in harnesses.values():
            h.close()

    manifest = dict(
        detector="patchcore",
        detector_choice_note=DETECTOR_CHOICE_NOTE,
        score_note=SCORE_NOTE,
        striking_case_note=STRIKING_CASE_NOTE,
        panel_order=PANEL_ORDER,
        panel_labels=PANEL_LABELS,
        striking_case_id=f"{scenes[0][0]}_{scenes[0][1]}_{scenes[0][2]}_sev{severities[-1]}",
        combos=[dict(id=c.combo_id, category=c.category, kind=c.kind, family=c.family,
                    severity=c.severity, panels=c.panels) for c in combos],
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    manifest = build_all()
    print(f"\n{len(manifest['combos'])} examples written to "
         f"{_demo_data_dir() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
