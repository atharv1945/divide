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
combination where classical Wiener destroys a scratch's structure (scratch
relative DRR 0.268, residual corr 0.024, aggregated across categories/
severities) while DIVIDE keeps it (relative DRR 1.189, residual corr 0.778)
- the project's single clearest side-by-side. Picked deliberately for that
reason; this specific example's own numbers (not the aggregate) were read
back from `demo_data/manifest.json` and checked against that expectation
before the demo shipped, not assumed to hold just because the aggregate
does. NOTE the word "destroys structure," not "erases": inspecting this
example directly during development corrected an earlier claim (see
FINDINGS.md Sec. 1) - Wiener's residual reaches almost the SAME peak
magnitude as the true residual, at essentially the same location. What's
gone is not the energy, it's the SHAPE: a uniformly-signed true residual
becomes sign-flipping ringing. See PROFILE_HALF_WIDTH / _extract_profile
below for the visualization that makes this legible.

A restored image alone does not make that comparison visible - a scratch is
a handful of pixels on a 256px image, so the four restored images can look
deceptively similar side by side even when their fidelity differs enormously
(see STRIKING_CASE_NOTE). Three additions fix that instead of picking a more
"dramatic" but less representative example: (1) each panel also gets its
residual map - (restored_a - restored_0), the exact quantity
defect_residual_correlation measures - rendered on a diverging colormap at
ONE shared, symmetric-around-zero scale across all four panels plus a
true-residual reference; (2) every restored image and residual map also
gets a zoomed crop around the defect mask's bounding box (with margin),
since the eye cannot find a 2-3px scratch at full frame regardless of how
it's colored; (3) - added after (1) and (2) turned out NOT to visually
separate Wiener from DIVIDE (both show a similarly-located, similarly-
bright mark: ringing concentrates at edges, same as a genuine defect does)
- a cross-defect residual PROFILE (a perpendicular slice through the
defect, true vs. every panel, one set of axes) and a per-pixel sign-
agreement map (green/red against the true residual's sign). These are what
actually separates them: DIVIDE's profile tracks true's clean single-signed
bump; Wiener's is spread into broad, off-center ringing lobes with no clean
peak at the true location.

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
from src.utils.paths import dtd_root, figures_dir, results_dir

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
    "The restored images AND the residual maps look deceptively similar - "
    "Wiener's ringing artifact concentrates right at the scratch's edge, "
    "reaching almost the same peak magnitude as the true residual, so a "
    "glance at either row can't tell the difference here. The profile "
    "plot above can: it's a slice perpendicular to the scratch, true "
    "residual vs. every panel's, on one axis with a zero line. True and "
    "DIVIDE both show a single clean bump centred at distance 0. Wiener "
    "shows no clean peak at the centre at all - its energy is smeared "
    "into broad ringing lobes on EITHER SIDE of the true location. "
    "The sign-agreement maps below show the same thing pixel by pixel: "
    "within the 524-pixel mask, the true residual is uniformly positive "
    "(+0.075 to +0.300) while Wiener's flips sign pixel to pixel (-0.034 "
    "to +0.296) - green where a panel's sign matches true's, red where "
    "it flips. Wiener isn't erasing the defect's energy - it's replacing "
    "its structure with ringing at the same location. See FINDINGS.md "
    "Sec. 1 for the corrected claim and the full evidence."
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


ZOOM_SIZE = 256      # zoomed crop is resized back up to this, so it reads at the same scale as the full panel
ZOOM_MARGIN_FRAC = 0.6
ZOOM_MIN_MARGIN = 20  # pixels, at IMAGE_SIZE=256 - a 2-3px scratch needs real headroom, not a tight box


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


def _save_rgb_u8(img_u8: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))


def _bbox_with_margin(mask: np.ndarray, size: int, margin_frac: float = ZOOM_MARGIN_FRAC,
                      min_margin: int = ZOOM_MIN_MARGIN) -> tuple[int, int, int, int]:
    """Defect mask's bounding box, expanded by margin_frac of its own size
    (with a pixel floor, since a 2-3px scratch's own size gives almost no
    margin otherwise) and clipped to the image. (y0, y1, x0, x1)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, size, 0, size
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    my = max(int((y1 - y0) * margin_frac), min_margin)
    mx = max(int((x1 - x0) * margin_frac), min_margin)
    return (max(0, y0 - my), min(size, y1 + my), max(0, x0 - mx), min(size, x1 + mx))


def _crop_zoom_img01(img01: np.ndarray, bbox: tuple[int, int, int, int], out_size: int = ZOOM_SIZE) -> np.ndarray:
    """Crop a float01 RGB image to bbox and resize back up to out_size -
    the zoom a full-frame image needs for a several-pixel defect to be
    visible at all (see module docstring point 2)."""
    y0, y1, x0, x1 = bbox
    crop_u8 = np.clip(img01[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(crop_u8, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def _crop_zoom_rgb_u8(img_u8: np.ndarray, bbox: tuple[int, int, int, int], out_size: int = ZOOM_SIZE) -> np.ndarray:
    """Same crop+zoom for an already-colorized (residual-map) uint8 RGB
    image - linear, not cubic, so the zoom doesn't ring/overshoot across
    the diverging colormap's hard-ish transitions."""
    y0, y1, x0, x1 = bbox
    crop = img_u8[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _residual_map(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    """Signed, single-channel residual (mean over RGB) - the exact
    quantity defect_residual_correlation compares against the true
    residual, just not yet reduced to a single correlation number."""
    return (a01 - b01).mean(axis=2)


def _residual_to_rgb(residual: np.ndarray, vmax: float) -> np.ndarray:
    """Diverging colormap (coolwarm), symmetric around zero at +/-vmax -
    vmax must be the SAME value across every panel being compared, passed
    in by the caller, never computed per-map here: per-map autoscaling is
    exactly what would make noise look like structure (module docstring
    point 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    vmax = max(float(vmax), 1e-6)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = matplotlib.colormaps["coolwarm"]
    rgba = cmap(norm(residual))
    return (rgba[..., :3] * 255).astype(np.uint8)


PROFILE_HALF_WIDTH = 20   # px each side of the scratch centerline
PROFILE_N_LINES = 5       # parallel cross-sections averaged together
PROFILE_COLORS = {"true": "black", "degraded": "#999999", "wiener": "#E69F00",
                  "restormer_deblur": "#56B4E9", "divide": "#009E73"}
SIGN_AGREE_COLOR = np.array([46, 160, 67], dtype=np.uint8)   # green
SIGN_DISAGREE_COLOR = np.array([214, 39, 40], dtype=np.uint8)  # red
SIGN_BG_COLOR = 230  # light gray, outside the mask


def _skeleton_path_order(mask: np.ndarray) -> np.ndarray:
    """Ordered (y, x) skeleton pixel coordinates tracing the defect mask's
    medial path - a greedy nearest-neighbor walk starting from a skeleton
    endpoint (a pixel with <=1 skeleton neighbor). Works for the thin,
    branchless strokes paste_anomaly draws (scratch/blob/texture masks);
    not general-purpose skeleton-graph tracing."""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask > 0)
    ys, xs = np.where(skel)
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    if len(pts) < 2:
        return pts

    def n_neighbors(i: int) -> int:
        d = np.abs(pts - pts[i]).max(axis=1)
        return int(((d <= 1) & (d > 0)).sum())

    counts = np.array([n_neighbors(i) for i in range(len(pts))])
    endpoints = np.where(counts <= 1)[0]
    start = int(endpoints[0]) if len(endpoints) else 0

    remaining = set(range(len(pts)))
    order = [start]
    remaining.discard(start)
    cur = start
    while remaining:
        rem_idx = np.array(list(remaining))
        d = np.sum((pts[rem_idx] - pts[cur]) ** 2, axis=1)
        nxt = int(rem_idx[np.argmin(d)])
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return pts[order]


def _tangent_normal_at(path: np.ndarray, idx: int, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Local tangent (finite difference over +/-window path steps) and its
    perpendicular normal, at path[idx]."""
    lo, hi = max(0, idx - window), min(len(path) - 1, idx + window)
    d = path[hi] - path[lo]
    n = np.linalg.norm(d)
    tangent = d / n if n > 1e-6 else np.array([1.0, 0.0])
    return tangent, np.array([-tangent[1], tangent[0]])


def _sample_along_normal(arr: np.ndarray, center: np.ndarray, normal: np.ndarray,
                         half_width: int = PROFILE_HALF_WIDTH) -> np.ndarray:
    """Bilinear-sampled 1D cross-section of `arr` through `center`, along
    `normal`, from -half_width to +half_width px."""
    from scipy.ndimage import map_coordinates

    offsets = np.arange(-half_width, half_width + 1)
    coords = np.stack([center[0] + offsets * normal[0], center[1] + offsets * normal[1]], axis=0)
    return map_coordinates(arr, coords, order=1, mode="constant", cval=np.nan)


def _extract_profile(mask: np.ndarray, residual: np.ndarray, n_lines: int = PROFILE_N_LINES,
                     half_width: int = PROFILE_HALF_WIDTH) -> np.ndarray:
    """Cross-defect residual profile: n_lines parallel normal cross-
    sections through the middle half of the mask's medial path, averaged
    together to reduce per-line noise. Falls back to a single horizontal
    cross-section through the mask centroid if the mask has too few
    skeleton pixels for a path (e.g. a blob/texture kind, not a thin
    stroke) - still a real, honestly-computed profile, just not
    tangent-aware for a shape with no clear tangent."""
    path = _skeleton_path_order(mask)
    if len(path) < 3:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.full(2 * half_width + 1, np.nan)
        center = np.array([ys.mean(), xs.mean()])
        return _sample_along_normal(residual, center, np.array([0.0, 1.0]), half_width)

    n = len(path)
    lo, hi = int(n * 0.25), int(n * 0.75)
    idxs = np.clip(np.linspace(lo, hi, n_lines).astype(int), 0, n - 1)
    profiles = [_sample_along_normal(residual, path[idx], _tangent_normal_at(path, idx)[1], half_width)
               for idx in idxs]
    return np.nanmean(np.stack(profiles), axis=0)


def _save_profile_figure(profiles: dict[str, np.ndarray], half_width: int, title: str, out_path: Path) -> None:
    """The textbook ringing visualization: cross-defect residual profiles
    for every panel plus the true residual, one set of axes, zero line.
    Genuine preservation reads as a single-signed bump matching true's;
    ringing reads as oscillation/broad elevation with no clean peak at
    the center."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(-half_width, half_width + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for key in ["true", "degraded", "wiener", "restormer_deblur", "divide"]:
        if key not in profiles:
            continue
        ax.plot(x, profiles[key], label=PANEL_LABELS.get(key, key),
               color=PROFILE_COLORS.get(key, "gray"), lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("distance from defect centerline (px)")
    ax.set_ylabel("residual value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _sign_agreement_rgb(panel_residual: np.ndarray, true_residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Green where sign(panel_residual) == sign(true_residual) within the
    defect mask, red where it flips - the exact quantity residual
    correlation is sensitive to, made spatially visible pixel by pixel
    rather than reduced to one number. Undefined (background gray)
    outside the mask - sign agreement isn't a meaningful question there."""
    h, w = panel_residual.shape
    rgb = np.full((h, w, 3), SIGN_BG_COLOR, dtype=np.uint8)
    m = mask > 0
    agree = np.sign(panel_residual) == np.sign(true_residual)
    rgb[m & agree] = SIGN_AGREE_COLOR
    rgb[m & ~agree] = SIGN_DISAGREE_COLOR
    return rgb


@dataclass
class Combo:
    combo_id: str
    category: str
    kind: str
    family: str
    severity: int
    panels: dict = field(default_factory=dict)
    true_residual: str = ""
    true_residual_zoom: str = ""
    profile: str = ""


def _panel_dict(image_path: str, image_zoom_path: str, heatmap_path: str,
                residual_path: str, residual_zoom_path: str, sign_agreement_path: str,
                score: float, mean_anomaly: float,
                rel_drr: float, residual_corr: float) -> dict:
    return dict(image=image_path, image_zoom=image_zoom_path, heatmap=heatmap_path,
               residual=residual_path, residual_zoom=residual_zoom_path,
               sign_agreement=sign_agreement_path,
               score=score, mean_anomaly=mean_anomaly,
               relative_drr=rel_drr, residual_corr=residual_corr)


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
            bbox = _bbox_with_margin(mask, size)

            for severity in severities:
                combo_id = f"{category}_{kind}_{family}_sev{severity}"
                y_a, y_0, _ = degrade_pair(x_a, x0, family, severity, seed=seed + scene_i)
                drr_identity = defect_retention_ratio(y_a, y_0, x_a, x0, mask)
                corr_identity = defect_residual_correlation(y_a, y_0, x_a, x0, mask)

                combo = Combo(combo_id, category, kind, family, severity)

                # Pass 1: every panel's restored image, score, metrics, and
                # RAW (unscaled) residual map - collected before any
                # colorizing, so the diverging colormap's scale can be set
                # ONCE from the true max across every panel plus the true
                # residual, not per panel (see module docstring point 1).
                true_residual = _residual_map(x_a, x0)
                raw = {}  # panel_key -> dict(image, score, mean_anomaly, rel_drr, residual_corr, residual)

                r_deg = harness.score(y_a)
                raw["degraded"] = dict(image=y_a, score=r_deg.score,
                                       mean_anomaly=float(np.mean(r_deg.anomaly_map)),
                                       rel_drr=1.0, residual_corr=corr_identity,
                                       residual=_residual_map(y_a, y_0))

                errors = {}
                for panel_key in ("wiener", "restormer_deblur", "divide"):
                    restorer = restorers[panel_key]
                    try:
                        r_a_img = restorer(y_a)
                        r_0_img = restorer(y_0)
                    except RuntimeError as e:
                        errors[panel_key] = str(e)
                        continue
                    drr = defect_retention_ratio(r_a_img, r_0_img, x_a, x0, mask)
                    rel = relative_drr(drr, drr_identity)
                    corr = defect_residual_correlation(r_a_img, r_0_img, x_a, x0, mask)
                    scored = harness.score(r_a_img)
                    raw[panel_key] = dict(image=r_a_img, score=scored.score,
                                          mean_anomaly=float(np.mean(scored.anomaly_map)),
                                          rel_drr=rel, residual_corr=corr,
                                          residual=_residual_map(r_a_img, r_0_img),
                                          anomaly_map=scored.anomaly_map)

                # shared symmetric colour scale across every panel AND the
                # true residual, so magnitude is directly comparable, not
                # just colour direction - a per-panel autoscale would make
                # a restorer's residual noise look exactly as "structured"
                # as DIVIDE's genuine scratch recovery.
                vmax = float(np.abs(true_residual).max())
                for v in raw.values():
                    vmax = max(vmax, float(np.abs(v["residual"]).max()))

                true_res_rgb = _residual_to_rgb(true_residual, vmax)
                true_res_p = img_dir / f"{combo_id}_true_residual.png"
                true_res_zoom_p = img_dir / f"{combo_id}_true_residual_zoom.png"
                _save_rgb_u8(true_res_rgb, true_res_p)
                _save_rgb_u8(_crop_zoom_rgb_u8(true_res_rgb, bbox), true_res_zoom_p)
                combo.true_residual = str(true_res_p.relative_to(data_dir))
                combo.true_residual_zoom = str(true_res_zoom_p.relative_to(data_dir))

                for panel_key in PANEL_ORDER:
                    if panel_key in errors:
                        combo.panels[panel_key] = dict(error=errors[panel_key])
                        continue
                    v = raw[panel_key]
                    img_p = img_dir / f"{combo_id}_{panel_key}.png"
                    img_zoom_p = img_dir / f"{combo_id}_{panel_key}_zoom.png"
                    heat_p = img_dir / f"{combo_id}_{panel_key}_heat.png"
                    res_p = img_dir / f"{combo_id}_{panel_key}_residual.png"
                    res_zoom_p = img_dir / f"{combo_id}_{panel_key}_residual_zoom.png"
                    sign_p = img_dir / f"{combo_id}_{panel_key}_sign_agreement.png"

                    _save_png(v["image"], img_p)
                    _save_rgb_u8(_crop_zoom_img01(v["image"], bbox), img_zoom_p)
                    amap = v.get("anomaly_map", r_deg.anomaly_map)
                    cv2.imwrite(str(heat_p), cv2.cvtColor(_colorize(amap), cv2.COLOR_RGB2BGR))
                    res_rgb = _residual_to_rgb(v["residual"], vmax)
                    _save_rgb_u8(res_rgb, res_p)
                    _save_rgb_u8(_crop_zoom_rgb_u8(res_rgb, bbox), res_zoom_p)
                    # sign agreement is only meaningful zoomed in on the mask -
                    # at full 256x256 frame it's a near-invisible dot, so only
                    # the zoomed crop is generated (module docstring point 3).
                    sign_rgb = _sign_agreement_rgb(v["residual"], true_residual, mask)
                    _save_rgb_u8(_crop_zoom_rgb_u8(sign_rgb, bbox), sign_p)

                    combo.panels[panel_key] = _panel_dict(
                        str(img_p.relative_to(data_dir)), str(img_zoom_p.relative_to(data_dir)),
                        str(heat_p.relative_to(data_dir)),
                        str(res_p.relative_to(data_dir)), str(res_zoom_p.relative_to(data_dir)),
                        str(sign_p.relative_to(data_dir)),
                        v["score"], v["mean_anomaly"], v["rel_drr"], v["residual_corr"])

                # Cross-defect residual profile - the textbook ringing
                # visualization (module docstring point 3): a perpendicular
                # slice through the defect, true residual vs. every panel's,
                # on one set of axes. This is the lead visual for the demo,
                # not an afterthought - generated for every combo so every
                # precomputed example has one, not just the striking case.
                profiles = {"true": _extract_profile(mask, true_residual)}
                for panel_key in PANEL_ORDER:
                    if panel_key in errors:
                        continue
                    profiles[panel_key] = _extract_profile(mask, raw[panel_key]["residual"])
                profile_p = img_dir / f"{combo_id}_profile.png"
                _save_profile_figure(profiles, PROFILE_HALF_WIDTH,
                                     f"Cross-defect residual profile: {category}/{kind}/{family} sev{severity}",
                                     profile_p)
                combo.profile = str(profile_p.relative_to(data_dir))

                if combo_id == f"{scenes[0][0]}_{scenes[0][1]}_{scenes[0][2]}_sev{severities[-1]}":
                    _save_profile_figure(
                        profiles, PROFILE_HALF_WIDTH,
                        "Cross-scratch residual profile - the project's clearest side-by-side",
                        figures_dir() / "striking_case_residual_profile.png")

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
                    severity=c.severity, panels=c.panels,
                    true_residual=c.true_residual, true_residual_zoom=c.true_residual_zoom,
                    profile=c.profile)
               for c in combos],
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
