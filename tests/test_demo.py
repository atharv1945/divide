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
    names = ["c1_degraded", "c1_degraded_heat", "c1_degraded_zoom", "c1_degraded_residual",
            "c1_degraded_residual_zoom", "c1_degraded_sign_agreement",
            "c1_wiener", "c1_wiener_heat", "c1_wiener_zoom",
            "c1_wiener_residual", "c1_wiener_residual_zoom", "c1_wiener_sign_agreement",
            "c1_true_residual", "c1_true_residual_zoom", "c1_profile"]
    for name in names:
        cv2.imwrite(str(img_dir / f"{name}.png"), img)

    def _panel(prefix, score, mean_anomaly, relative_drr, residual_corr):
        return dict(image=f"images/{prefix}.png", image_zoom=f"images/{prefix}_zoom.png",
                   heatmap=f"images/{prefix}_heat.png", residual=f"images/{prefix}_residual.png",
                   residual_zoom=f"images/{prefix}_residual_zoom.png",
                   sign_agreement=f"images/{prefix}_sign_agreement.png",
                   score=score, mean_anomaly=mean_anomaly,
                   relative_drr=relative_drr, residual_corr=residual_corr)

    manifest = dict(
        detector="patchcore",
        detector_choice_note="PatchCore chosen because ... (test note)",
        score_note="score saturates because ... (test note)",
        striking_case_note="look at the profile plot, not just the mark (test note)",
        panel_order=["degraded", "wiener", "restormer_deblur", "divide"],
        panel_labels=precompute.PANEL_LABELS,
        striking_case_id="c1",
        combos=[dict(
            id="c1", category="bottle", kind="scratch", family="defocus", severity=3,
            true_residual="images/c1_true_residual.png",
            true_residual_zoom="images/c1_true_residual_zoom.png",
            profile="images/c1_profile.png",
            panels=dict(
                degraded=_panel("c1_degraded", 0.5, 0.31, 1.0, 0.6),
                wiener=_panel("c1_wiener", 0.4, 0.22, 0.27, 0.02),
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


def test_panel_text_leads_with_mean_anomaly_not_score():
    """The calibrated score saturates to 1.000 everywhere in this demo's
    data-scarce regime (see SCORE_NOTE) - it must be demoted to a small
    secondary line, not lead the panel, or four identical 1.000s read as
    'nothing differs', the opposite of the finding."""
    from src.demo.app import _panel_text
    text = _panel_text(dict(score=0.4231, mean_anomaly=0.31, relative_drr=0.268, residual_corr=0.024))
    assert "0.4231" in text and "0.268" in text and "0.024" in text and "0.31" in text
    assert text.index("mean anomaly") < text.index("score")


def test_panel_text_formats_error():
    from src.demo.app import _panel_text
    text = _panel_text(dict(error="checkpoint not found"))
    assert "UNAVAILABLE" in text and "checkpoint not found" in text


def test_panel_paths_none_for_error_panel(tmp_path):
    from src.demo.app import PANEL_IMAGE_FIELDS, _panel_paths
    paths = _panel_paths(tmp_path, dict(error="no weights"))
    assert set(paths) == set(PANEL_IMAGE_FIELDS)
    assert all(v is None for v in paths.values())


def test_panel_paths_resolves_all_image_fields_against_data_dir(tmp_path):
    from src.demo.app import PANEL_IMAGE_FIELDS, _panel_paths
    panel = {f: f"images/x_{f}.png" for f in PANEL_IMAGE_FIELDS}
    paths = _panel_paths(tmp_path, panel)
    for f in PANEL_IMAGE_FIELDS:
        assert paths[f] == str(tmp_path / f"images/x_{f}.png")


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


def test_build_html_embeds_residual_and_zoom_images(tmp_path, monkeypatch):
    """Regression test for the residual-map/zoom addition - a viewer must
    get the residual map and zoomed crops, not just the restored image."""
    from src.demo.app import load_manifest
    from src.demo.static_demo import build_html

    data_dir, _ = _write_fake_manifest(tmp_path, monkeypatch)
    html = build_html(load_manifest(), data_dir)
    # 2 real (non-error) panels x 6 embedded images (image, image_zoom,
    # heatmap, residual, residual_zoom, sign_agreement) + 2 combo-level
    # true-residual images + 1 combo-level profile plot
    assert html.count("data:image/png;base64,") == 2 * 6 + 2 + 1
    assert "true_residual" in html or "true residual" in html.lower()
    assert "sign agreement" in html.lower()
    assert "residual profile" in html.lower()


# --------------------------------------------------------------------------
# precompute.py - residual map / zoom-crop helpers
# --------------------------------------------------------------------------

def test_bbox_with_margin_expands_and_clips_to_image():
    from src.demo.precompute import _bbox_with_margin
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 90:98] = 1  # near the right edge, so margin must clip, not overflow
    y0, y1, x0, x1 = _bbox_with_margin(mask, size=100, margin_frac=0.5, min_margin=10)
    assert 0 <= y0 < 45 and y1 > 55 and y1 <= 100
    assert 0 <= x0 < 90 and x1 == 100  # clipped, not 98 + margin


def test_bbox_with_margin_full_frame_for_empty_mask():
    from src.demo.precompute import _bbox_with_margin
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert _bbox_with_margin(mask, size=64) == (0, 64, 0, 64)


def test_residual_map_is_signed_and_zero_for_identical_images():
    from src.demo.precompute import _residual_map
    a = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert np.allclose(_residual_map(a, a), 0.0)

    b = a.copy()
    b[0, 0] = 0.9  # brighter here
    r = _residual_map(b, a)
    assert r[0, 0] > 0
    assert np.allclose(r[1:], 0.0)


def test_residual_to_rgb_uses_the_passed_vmax_not_the_maps_own_range():
    """The whole point of a shared scale: two maps with very different
    magnitudes, colorized at the SAME vmax, must not both look saturated -
    only the one that actually reaches vmax should hit the colormap's
    extreme colour."""
    from src.demo.precompute import _residual_to_rgb
    big = np.full((4, 4), 1.0, dtype=np.float32)
    small = np.full((4, 4), 0.05, dtype=np.float32)
    rgb_big = _residual_to_rgb(big, vmax=1.0)
    rgb_small = _residual_to_rgb(small, vmax=1.0)
    assert not np.array_equal(rgb_big, rgb_small)
    # small, at 5% of vmax, must be close to the neutral (zero) colour -
    # not stretched to look like a large residual.
    neutral = _residual_to_rgb(np.zeros((4, 4), dtype=np.float32), vmax=1.0)
    assert np.abs(rgb_small.astype(int) - neutral.astype(int)).max() < 40


def test_crop_zoom_img01_resizes_crop_to_out_size():
    from src.demo.precompute import _crop_zoom_img01
    img = np.random.default_rng(0).random((100, 100, 3)).astype(np.float32)
    out = _crop_zoom_img01(img, bbox=(10, 30, 10, 30), out_size=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_crop_zoom_rgb_u8_resizes_crop_to_out_size():
    from src.demo.precompute import _crop_zoom_rgb_u8
    img = (np.random.default_rng(0).random((100, 100, 3)) * 255).astype(np.uint8)
    out = _crop_zoom_rgb_u8(img, bbox=(10, 30, 10, 30), out_size=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------
# precompute.py - cross-defect profile / sign-agreement (residual-map
# follow-up: the colorized maps alone didn't visually separate a restorer
# that destroys structure from one that preserves it - see module
# docstring point 3)
# --------------------------------------------------------------------------

def _straight_line_mask(size=64, width=2):
    """A short straight diagonal stroke - skeletonize()-friendly, unlike a
    single-pixel dot, and with an unambiguous tangent direction so the
    profile's normal can be checked against a known geometry."""
    import cv2
    m = np.zeros((size, size), dtype=np.uint8)
    cv2.line(m, (20, 20), (44, 44), 1, width)
    return m


def test_skeleton_path_order_traces_a_straight_stroke_end_to_end():
    from src.demo.precompute import _skeleton_path_order
    mask = _straight_line_mask()
    path = _skeleton_path_order(mask)
    assert len(path) > 5
    # a straight diagonal stroke's ordered path should move monotonically
    # from one end to the other, not jump around
    d = np.diff(path, axis=0)
    step_lengths = np.linalg.norm(d, axis=1)
    assert step_lengths.max() < 3  # consecutive path points stay close together


def test_skeleton_path_order_handles_degenerate_mask():
    from src.demo.precompute import _skeleton_path_order
    empty = np.zeros((32, 32), dtype=np.uint8)
    assert len(_skeleton_path_order(empty)) == 0
    single = np.zeros((32, 32), dtype=np.uint8)
    single[10, 10] = 1
    path = _skeleton_path_order(single)
    assert len(path) <= 1


def test_tangent_normal_at_is_unit_and_perpendicular():
    from src.demo.precompute import _tangent_normal_at
    path = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    tangent, normal = _tangent_normal_at(path, idx=2, window=2)
    assert np.isclose(np.linalg.norm(tangent), 1.0)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.dot(tangent, normal), 0.0)  # perpendicular


def test_extract_profile_peaks_at_the_defect_for_a_true_residual():
    """The textbook case this whole visualization exists for: a residual
    that's genuinely localized AT the defect must show a clean, centered
    peak in its cross-section - not spread out, not offset."""
    from src.demo.precompute import _extract_profile
    mask = _straight_line_mask(size=64, width=2)
    residual = mask.astype(np.float32) * 0.5   # residual = 0.5 exactly where the mask is
    profile = _extract_profile(mask, residual, n_lines=3, half_width=10)
    assert len(profile) == 21
    center = profile[10]
    edges = np.concatenate([profile[:5], profile[-5:]])
    assert center > np.nanmax(edges) + 0.1  # a real, not marginal, peak at the center


def test_extract_profile_falls_back_for_a_mask_with_no_clean_skeleton_path():
    """blob/texture masks aren't thin strokes - _extract_profile must still
    return a usable profile (centroid cross-section), not crash."""
    from src.demo.precompute import _extract_profile
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 1  # a filled blob, not a stroke
    residual = mask.astype(np.float32) * 0.3
    profile = _extract_profile(mask, residual, half_width=10)
    assert len(profile) == 21
    assert np.nanmax(profile) > 0


def test_sign_agreement_rgb_marks_matches_green_and_flips_red():
    from src.demo.precompute import SIGN_AGREE_COLOR, SIGN_DISAGREE_COLOR, _sign_agreement_rgb
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    true_res = np.full((4, 4), 0.2, dtype=np.float32)          # positive everywhere
    panel_res = np.full((4, 4), 0.2, dtype=np.float32)
    panel_res[1, 1] = -0.1                                      # one flipped pixel, inside the mask

    rgb = _sign_agreement_rgb(panel_res, true_res, mask)
    assert np.array_equal(rgb[1, 2], SIGN_AGREE_COLOR)     # in mask, sign matches
    assert np.array_equal(rgb[1, 1], SIGN_DISAGREE_COLOR)  # in mask, sign flipped
    assert not np.array_equal(rgb[0, 0], SIGN_AGREE_COLOR)     # outside mask - neutral, not green
    assert not np.array_equal(rgb[0, 0], SIGN_DISAGREE_COLOR)  # outside mask - neutral, not red


def test_save_profile_figure_writes_a_file(tmp_path):
    from src.demo.precompute import _save_profile_figure
    profiles = {"true": np.linspace(-1, 1, 21), "wiener": np.random.default_rng(0).normal(size=21)}
    out = tmp_path / "profile.png"
    _save_profile_figure(profiles, half_width=10, title="test", out_path=out)
    assert out.exists() and out.stat().st_size > 0
