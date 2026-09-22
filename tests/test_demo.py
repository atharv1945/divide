import json

import numpy as np
import pytest


def _write_fake_manifest(tmp_path, monkeypatch):
    """Builds a tiny, structurally-real manifest + PNGs so app.py/
    static_demo.py can be tested without running precompute.py's real
    (slow, model-dependent) pipeline."""
    import cv2

    import src.demo.app as app
    import src.demo.precompute as precompute

    data_dir = tmp_path / "demo_data"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True)
    monkeypatch.setattr(precompute, "_demo_data_dir", lambda: data_dir)
    monkeypatch.setattr(app, "_demo_data_dir", lambda: data_dir)

    img = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
    for name in ["c1_degraded", "c1_degraded_heat", "c1_wiener", "c1_wiener_heat"]:
        cv2.imwrite(str(img_dir / f"{name}.png"), img)

    manifest = dict(
        detector="patchcore",
        detector_choice_note="PatchCore chosen because ... (test note)",
        score_note="score saturates because ... (test note)",
        striking_case_note="look at the correlation, not just the mark (test note)",
        panel_order=["degraded", "wiener", "restormer_deblur", "divide"],
        panel_labels=precompute.PANEL_LABELS,
        striking_case_id="c1",
        combos=[dict(
            id="c1", category="bottle", kind="scratch", family="defocus", severity=3,
            panels=dict(
                degraded=dict(image="images/c1_degraded.png", heatmap="images/c1_degraded_heat.png",
                             score=0.5, mean_anomaly=0.31, relative_drr=1.0, residual_corr=0.6),
                wiener=dict(image="images/c1_wiener.png", heatmap="images/c1_wiener_heat.png",
                           score=0.4, mean_anomaly=0.22, relative_drr=0.27, residual_corr=0.02),
                restormer_deblur=dict(error="weights not found"),
                divide=dict(error="checkpoint not found"),
            ),
        )],
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    return data_dir, manifest


# --------------------------------------------------------------------------
# app.py - pure lookup, no model/detector dependencies to import-skip on
# --------------------------------------------------------------------------

def test_load_manifest_fails_loudly_when_missing(tmp_path, monkeypatch):
    import src.demo.app as app
    monkeypatch.setattr(app, "_demo_data_dir", lambda: tmp_path / "nope")

    with pytest.raises(RuntimeError) as exc:
        app.load_manifest()
    assert "src.demo.precompute" in str(exc.value)


def test_load_manifest_and_combo_lookup(tmp_path, monkeypatch):
    import src.demo.app as app
    _write_fake_manifest(tmp_path, monkeypatch)

    manifest = app.load_manifest()
    assert app.combo_choices(manifest) == ["c1"]
    combo = app.find_combo(manifest, "c1")
    assert combo["category"] == "bottle"

    with pytest.raises(KeyError):
        app.find_combo(manifest, "does-not-exist")


def test_panel_text_formats_score_drr_corr():
    from src.demo.app import _panel_text
    text = _panel_text(dict(score=0.4231, mean_anomaly=0.31, relative_drr=0.268, residual_corr=0.024))
    assert "0.4231" in text and "0.268" in text and "0.024" in text and "0.31" in text


def test_panel_text_formats_error():
    from src.demo.app import _panel_text
    text = _panel_text(dict(error="checkpoint not found"))
    assert "UNAVAILABLE" in text and "checkpoint not found" in text


def test_panel_paths_none_for_error_panel(tmp_path, monkeypatch):
    from src.demo.app import _panel_paths
    img, heat = _panel_paths(tmp_path, dict(error="no weights"))
    assert img is None and heat is None


def test_panel_paths_resolves_against_data_dir(tmp_path):
    from src.demo.app import _panel_paths
    panel = dict(image="images/x.png", heatmap="images/x_heat.png")
    img, heat = _panel_paths(tmp_path, panel)
    assert img == str(tmp_path / "images/x.png")
    assert heat == str(tmp_path / "images/x_heat.png")


def test_build_demo_constructs_blocks_without_launching(tmp_path, monkeypatch):
    """Catches Gradio API breakage without opening a server/port, using a
    fake precomputed manifest so this doesn't depend on real model weights
    or a real precompute run."""
    pytest.importorskip("gradio")
    _write_fake_manifest(tmp_path, monkeypatch)

    from src.demo.app import build_demo
    demo = build_demo()
    assert demo is not None


# --------------------------------------------------------------------------
# static_demo.py
# --------------------------------------------------------------------------

def test_build_html_embeds_images_and_striking_case(tmp_path, monkeypatch):
    from src.demo.app import load_manifest
    from src.demo.static_demo import build_html

    data_dir, _ = _write_fake_manifest(tmp_path, monkeypatch)
    manifest = load_manifest()
    html = build_html(manifest, data_dir)

    assert "data:image/png;base64," in html
    assert "c1" in html
    assert "checkpoint not found" in html  # divide's error surfaces, not silently dropped
    assert "PatchCore chosen because" in html  # detector_choice_note present verbatim


def test_build_html_is_self_contained_single_file(tmp_path, monkeypatch):
    """No external script/style references - must work with no network and
    no other files present besides the one HTML file."""
    from src.demo.app import load_manifest
    from src.demo.static_demo import build_html

    data_dir, _ = _write_fake_manifest(tmp_path, monkeypatch)
    html = build_html(load_manifest(), data_dir)
    assert "<script src=" not in html
    assert "<link " not in html
