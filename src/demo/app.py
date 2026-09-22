"""Gradio demo: degraded / classical Wiener / Restormer deblur / DIVIDE,
side by side. The LEAD visual is the cross-defect residual profile - a
perpendicular slice through the defect, true residual vs. every panel's,
on one set of axes - because the restored images and even the colorized
residual maps alone do not visually separate a restorer that destroys the
defect's structure (ringing, same location, comparable peak magnitude)
from one that genuinely preserves it (see FINDINGS.md Sec. 1). Below that:
each panel's restored image (full + zoomed crop), PatchCore heatmap,
residual map (restored_a - restored_0, full + zoomed crop, one shared
colour scale with the true-residual reference), a sign-agreement map
(green/red against the true residual's sign - the exact thing residual
correlation measures, made spatially visible), and the numbers: mean
anomaly first, then relative DRR and residual correlation, with the
saturated calibrated score demoted to a small secondary line.

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

PANEL_IMAGE_FIELDS = ["image", "image_zoom", "heatmap", "residual", "residual_zoom", "sign_agreement"]


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
    """mean_anomaly leads (it's the number that actually varies here);
    the calibrated 'score' - which saturates to 1.000 for every panel in
    this data-scarce regime, see SCORE_NOTE - is a small secondary line,
    not the headline. Four identical 1.000s side by side used to read as
    'nothing differs', the opposite of the finding."""
    if "error" in panel:
        return f"UNAVAILABLE:\n{panel['error']}"
    return (f"mean anomaly = {panel['mean_anomaly']:.4f}\n"
           f"relative DRR = {panel['relative_drr']:.3f}\n"
           f"residual corr = {panel['residual_corr']:.3f}\n"
           f"(score = {panel['score']:.4f}, calibrated - saturates here, see note above)")


def _panel_paths(data_dir: Path, panel: dict) -> dict[str, str | None]:
    if "error" in panel:
        return {f: None for f in PANEL_IMAGE_FIELDS}
    return {f: str(data_dir / panel[f]) for f in PANEL_IMAGE_FIELDS}


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
        out = [str(data_dir / combo["profile"])]
        for key in PANEL_ORDER:
            panel = combo["panels"].get(key, {"error": "not computed"})
            paths = _panel_paths(data_dir, panel)
            out.extend([paths[f] for f in PANEL_IMAGE_FIELDS])
            out.append(_panel_text(panel))
        out.append(str(data_dir / combo["true_residual"]))
        out.append(str(data_dir / combo["true_residual_zoom"]))
        scene = f"{combo['category']} / {combo['kind']} defect / {combo['family']} severity {combo['severity']}"
        if combo_id == manifest.get("striking_case_id"):
            scene += f"\n\n**This is the project's clearest side-by-side.** {manifest['striking_case_note']}"
        out.append(scene)
        return out

    with gr.Blocks(title="DIVIDE demo") as demo:
        gr.Markdown(
            "# DIVIDE - does inverse-degradation restoration help detection?\n"
            "Precomputed only: every image, heatmap, residual map, profile, "
            "score, relative DRR, and residual correlation below was "
            "computed once by `src.demo.precompute`, not live. Picking an "
            "example is a dictionary lookup.\n\n"
            f"**Detector: PatchCore.** {manifest['detector_choice_note']}\n\n"
            f"**On the 'score' number:** {manifest['score_note']}"
        )
        combo_dd = gr.Dropdown(choices=choices, value=default_id,
                               label="precomputed example (category / defect kind / family / severity)")
        scene_label = gr.Markdown()

        gr.Markdown(
            "## Cross-defect residual profile\n"
            "**Start here.** A slice perpendicular to the defect, true "
            "residual vs. every panel's, on one axis - the textbook way to "
            "see whether a restorer relocated the defect's energy into "
            "ringing or genuinely recovered it. The images below are "
            "supporting detail, not the headline."
        )
        profile_img = gr.Image(label="cross-defect residual profile")

        gr.Markdown(
            "**True residual (clean_with_defect - clean_without_defect)** - "
            "the ground truth every panel's own residual is compared "
            "against below. Same colour scale as every panel."
        )
        with gr.Row():
            true_res_img = gr.Image(label="true residual")
            true_res_zoom_img = gr.Image(label="true residual (zoomed)")

        panel_images, panel_zooms, panel_heatmaps = [], [], []
        panel_residuals, panel_residual_zooms, panel_signs, panel_texts = [], [], [], []
        with gr.Row():
            for key in PANEL_ORDER:
                with gr.Column():
                    gr.Markdown(f"**{manifest['panel_labels'][key]}**")
                    panel_images.append(gr.Image(label="restored image"))
                    panel_zooms.append(gr.Image(label="restored, zoomed to defect"))
                    panel_heatmaps.append(gr.Image(label="PatchCore heatmap"))
                    panel_residuals.append(gr.Image(label="residual (restored_a - restored_0)"))
                    panel_residual_zooms.append(gr.Image(label="residual, zoomed to defect"))
                    panel_signs.append(gr.Image(label="sign agreement vs. true (green=match, red=flip)"))
                    panel_texts.append(gr.Textbox(label="numbers", interactive=False, lines=4))

        outputs = [profile_img]
        for img_c, zoom_c, heat_c, res_c, resz_c, sign_c, txt_c in zip(
                panel_images, panel_zooms, panel_heatmaps,
                panel_residuals, panel_residual_zooms, panel_signs, panel_texts):
            outputs += [img_c, zoom_c, heat_c, res_c, resz_c, sign_c, txt_c]
        outputs += [true_res_img, true_res_zoom_img, scene_label]

        combo_dd.change(on_change, [combo_dd], outputs)
        demo.load(on_change, [combo_dd], outputs)

    return demo


def launch() -> None:
    build_demo().launch()


if __name__ == "__main__":
    launch()
