"""Gradio demo: clean / degraded / degraded+Restormer / degraded+DIVIDE.

Four panels side by side, each showing the image, a detector heatmap
overlay, and the anomaly score. Sliders pick degradation family and
severity; a live relative-DRR readout sits under panels 3 and 4.

The example image carries a synthetic pasted anomaly with a known mask, so
DRR is actually computable live (real MVTec test defects have no
defect-free counterpart to diff against - see degrade_pair()'s docstring).

The frozen detector is fit ONCE at import time and reused for every slider
move ("cache the detector memory bank so inference is instant" - this is
what DetectorHarness.fit() already guarantees; the demo just has to avoid
re-fitting on every callback).

Restormer and DIVIDE both need weights/checkpoints this machine doesn't
have. Rather than crash, their panels show the exact fail-loud message in
place of an image - this is what "present it live" will actually look like
on a laptop with nothing downloaded, and it's the honest state to ship:
wiring verified, weights not.

    python -m src.demo.app
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.data.mvtec import synthetic_split
from src.degrade.anomaly import paste_anomaly
from src.degrade.simulator import FAMILIES, SEVERITIES, degrade_pair
from src.detect.harness import DetectorHarness
from src.metrics.core import defect_retention_ratio, relative_drr
from src.models.restorers import get_restorer

CATEGORY = "carpet"
IMAGE_SIZE = 64  # multiple of 32 - see harness.py on ReverseDistillation's constraint
DETECTOR_NAME = "padim"  # the one reliably verified in this environment
EXAMPLE_SEED = 0


@dataclass
class PanelResult:
    image: np.ndarray            # (H,W,3) float32 in [0,1], possibly heatmap-overlaid
    score: float | None
    error: str | None = None
    drr_rel: float | None = None


def _colorize_heatmap(amap: np.ndarray) -> np.ndarray:
    a = amap - amap.min()
    a = a / (a.max() + 1e-8)
    u8 = (a * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def overlay_heatmap(img: np.ndarray, amap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = _colorize_heatmap(amap)
    if heat.shape[:2] != img.shape[:2]:
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1).astype(np.float32)


class DemoState:
    """Built once at import/startup, reused across every slider callback."""

    def __init__(self, category: str = CATEGORY, size: int = IMAGE_SIZE,
                detector_name: str = DETECTOR_NAME):
        train, test = synthetic_split(category, n_train=16, n_test_good=4,
                                      n_test_bad=1, size=size, seed=EXAMPLE_SEED)
        self.x0 = train[0].image  # clean, defect-free example
        rng = np.random.default_rng(EXAMPLE_SEED)
        self.x_a, self.mask, self.spec = paste_anomaly(self.x0, rng, kind="scratch")

        self.harness = DetectorHarness(detector_name, image_size=size, seed=EXAMPLE_SEED)
        self.harness.fit([s.image for s in train])

        self._restorer_cache: dict[str, object] = {}

    def restorer(self, name: str):
        if name not in self._restorer_cache:
            self._restorer_cache[name] = get_restorer(name)
        return self._restorer_cache[name]

    def close(self) -> None:
        self.harness.close()


_STATE: DemoState | None = None


def get_state() -> DemoState:
    global _STATE
    if _STATE is None:
        _STATE = DemoState()
    return _STATE


def _score_panel(state: DemoState, img: np.ndarray) -> PanelResult:
    r = state.harness.score(img)
    return PanelResult(image=overlay_heatmap(img, r.anomaly_map), score=r.score)


def _restored_panel(state: DemoState, restorer_name: str, y_a: np.ndarray,
                    y_0: np.ndarray, drr_identity: float) -> PanelResult:
    try:
        restorer = state.restorer(restorer_name)
        r_a = restorer(y_a)
        r_0 = restorer(y_0)
    except RuntimeError as e:
        return PanelResult(image=np.zeros_like(y_a), score=None, error=str(e))

    drr = defect_retention_ratio(r_a, r_0, state.x_a, state.x0, state.mask)
    rel = relative_drr(drr, drr_identity)
    scored = state.harness.score(r_a)
    return PanelResult(image=overlay_heatmap(r_a, scored.anomaly_map),
                       score=scored.score, drr_rel=rel)


def build_panels(family: str, severity: int) -> dict[str, PanelResult]:
    """Pure function: (family, severity) -> the four panels. No Gradio here -
    this is what test_demo.py exercises directly."""
    state = get_state()
    y_a, y_0, _ = degrade_pair(state.x_a, state.x0, family, severity, seed=EXAMPLE_SEED)

    drr_identity = defect_retention_ratio(y_a, y_0, state.x_a, state.x0, state.mask)

    return {
        "clean": _score_panel(state, state.x_a),
        "degraded": _score_panel(state, y_a),
        "restormer": _restored_panel(state, "restormer", y_a, y_0, drr_identity),
        "divide": _restored_panel(state, "divide", y_a, y_0, drr_identity),
    }


# --------------------------------------------------------------------------
# Gradio wiring
# --------------------------------------------------------------------------

def _panel_outputs(p: PanelResult) -> tuple:
    img_u8 = (np.clip(p.image, 0, 1) * 255).astype(np.uint8)
    if p.error is not None:
        text = f"UNAVAILABLE:\n{p.error}"
    else:
        text = f"score = {p.score:.4f}" if p.score is not None else ""
        if p.drr_rel is not None:
            text += f"\nrelative DRR = {p.drr_rel:.3f}"
    return img_u8, text


def build_demo():
    """Construct the gr.Blocks app without launching a server - this is what
    test_demo.py imports to catch Gradio API breakage (requirements.txt
    says gradio>=4.0; this environment has 6.28.0, a large version gap)."""
    import gradio as gr

    def on_change(family: str, severity: int):
        panels = build_panels(family, int(severity))
        out = []
        for key in ("clean", "degraded", "restormer", "divide"):
            out.extend(_panel_outputs(panels[key]))
        return out

    with gr.Blocks(title="DIVIDE demo") as demo:
        gr.Markdown(
            "# DIVIDE - inverse degradation vs. free-form restoration\n"
            "Detector is frozen PaDiM, fit once on clean normals. "
            "Restormer/DIVIDE panels show the exact fail-loud message when "
            "their weights aren't present."
        )
        with gr.Row():
            family_dd = gr.Dropdown(choices=FAMILIES, value="defocus", label="family")
            severity_sl = gr.Slider(minimum=min(SEVERITIES), maximum=max(SEVERITIES),
                                    step=1, value=3, label="severity")

        panel_images, panel_texts = [], []
        with gr.Row():
            for title in ["clean", "degraded", "degraded + Restormer", "degraded + DIVIDE"]:
                with gr.Column():
                    gr.Markdown(f"**{title}**")
                    panel_images.append(gr.Image(label="detector heatmap"))
                    panel_texts.append(gr.Textbox(label="score", interactive=False))

        outputs = []
        for img_c, txt_c in zip(panel_images, panel_texts):
            outputs += [img_c, txt_c]

        family_dd.change(on_change, [family_dd, severity_sl], outputs)
        severity_sl.change(on_change, [family_dd, severity_sl], outputs)
        demo.load(on_change, [family_dd, severity_sl], outputs)

    return demo


def launch() -> None:
    build_demo().launch()


if __name__ == "__main__":
    launch()
