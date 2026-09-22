"""Gradio demo: degraded / classical Wiener / Restormer deblur / DIVIDE,
side by side, each panel showing the restored image, the PatchCore anomaly
heatmap, the detection score, relative DRR, and residual correlation.

Precomputed only - this app does nothing but look up rows in
demo_data/manifest.json and load the PNGs beside it. No detector, no
restorer, no model runs when a slider moves; everything was computed once
by `src.demo.precompute`. Run that first if demo_data/manifest.json is
missing - this module fails loudly, naming that exact command, rather than
falling back to live inference.

    python -m src.demo.precompute   # once, offline - a few minutes
    python -m src.demo.app          # instant, lookups only
"""
from __future__ import annotations

import json
from pathlib import Path

from src.demo.precompute import PANEL_ORDER, _demo_data_dir


def _manifest_path() -> Path:
    return _demo_data_dir() / "manifest.json"


def load_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        raise RuntimeError(
            f"\n{'=' * 70}\n"
            f"No precomputed demo data at {path}.\n"
            f"This demo is lookup-only by design - it does not run models.\n"
            f"Generate the data first:\n"
            f"  python -m src.demo.precompute\n"
            f"{'=' * 70}"
        )
    return json.loads(path.read_text())


def combo_choices(manifest: dict) -> list[str]:
    return [c["id"] for c in manifest["combos"]]


def find_combo(manifest: dict, combo_id: str) -> dict:
    for c in manifest["combos"]:
        if c["id"] == combo_id:
            return c
    raise KeyError(f"no such precomputed example: {combo_id!r}")


def _panel_text(panel: dict) -> str:
    if "error" in panel:
        return f"UNAVAILABLE:\n{panel['error']}"
    return (f"score = {panel['score']:.4f}  (mean anomaly = {panel['mean_anomaly']:.4f})\n"
           f"relative DRR = {panel['relative_drr']:.3f}\n"
           f"residual corr = {panel['residual_corr']:.3f}")


def _panel_paths(data_dir: Path, panel: dict) -> tuple[str | None, str | None]:
    if "error" in panel:
        return None, None
    return str(data_dir / panel["image"]), str(data_dir / panel["heatmap"])


def build_demo():
    """Construct the gr.Blocks app without launching a server."""
    import gradio as gr

    manifest = load_manifest()
    data_dir = _demo_data_dir()
    choices = combo_choices(manifest)
    default_id = manifest.get("striking_case_id", choices[0])
    if default_id not in choices:
        default_id = choices[0]

    def on_change(combo_id: str):
        combo = find_combo(manifest, combo_id)
        out = []
        for key in PANEL_ORDER:
            panel = combo["panels"].get(key, {"error": "not computed"})
            img_path, heat_path = _panel_paths(data_dir, panel)
            out.extend([img_path, heat_path, _panel_text(panel)])
        scene = f"{combo['category']} / {combo['kind']} defect / {combo['family']} severity {combo['severity']}"
        if combo_id == manifest.get("striking_case_id"):
            scene += f"\n\n**This is the project's clearest side-by-side.** {manifest['striking_case_note']}"
        out.append(scene)
        return out

    with gr.Blocks(title="DIVIDE demo") as demo:
        gr.Markdown(
            "# DIVIDE - does inverse-degradation restoration help detection?\n"
            "Precomputed only: every image, heatmap, score, relative DRR, and "
            "residual correlation below was computed once by "
            "`src.demo.precompute`, not live. Picking an example is a "
            "dictionary lookup.\n\n"
            f"**Detector: PatchCore.** {manifest['detector_choice_note']}\n\n"
            f"**On the 'score' number:** {manifest['score_note']}"
        )
        combo_dd = gr.Dropdown(choices=choices, value=default_id,
                               label="precomputed example (category / defect kind / family / severity)")
        scene_label = gr.Markdown()

        panel_images, panel_heatmaps, panel_texts = [], [], []
        with gr.Row():
            for key in PANEL_ORDER:
                with gr.Column():
                    gr.Markdown(f"**{manifest['panel_labels'][key]}**")
                    panel_images.append(gr.Image(label="restored image"))
                    panel_heatmaps.append(gr.Image(label="PatchCore heatmap"))
                    panel_texts.append(gr.Textbox(label="score / DRR / corr", interactive=False, lines=3))

        outputs = []
        for img_c, heat_c, txt_c in zip(panel_images, panel_heatmaps, panel_texts):
            outputs += [img_c, heat_c, txt_c]
        outputs.append(scene_label)

        combo_dd.change(on_change, [combo_dd], outputs)
        demo.load(on_change, [combo_dd], outputs)

    return demo


def launch() -> None:
    build_demo().launch()


if __name__ == "__main__":
    launch()
