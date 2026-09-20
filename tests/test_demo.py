import numpy as np
import pytest

pytest.importorskip("anomalib")
pytest.importorskip("lightning")
pytest.importorskip("gradio")


def test_build_panels_runs_and_shapes_are_consistent():
    from src.demo.app import build_panels, IMAGE_SIZE

    panels = build_panels("noise", 3)
    assert set(panels) == {"clean", "degraded", "restormer", "divide"}

    for key, p in panels.items():
        assert p.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert np.isfinite(p.image).all()
        assert p.image.min() >= 0.0 and p.image.max() <= 1.0


def test_clean_and_degraded_panels_always_score_successfully():
    from src.demo.app import build_panels

    panels = build_panels("defocus", 4)
    assert panels["clean"].score is not None
    assert panels["clean"].error is None
    assert panels["degraded"].score is not None
    assert panels["degraded"].error is None


def test_divide_panel_fails_loud_not_crash_without_weights():
    """'divide' has no trained checkpoint in this environment (or any CI
    environment - it's this project's own model, nothing to download) - the
    demo must show the fail-loud message in that panel, not raise out of
    build_panels()."""
    from src.demo.app import build_panels

    panels = build_panels("illumination", 2)
    p = panels["divide"]
    assert p.error is not None
    assert p.score is None
    assert np.array_equal(p.image, np.zeros_like(p.image))


def test_restormer_panel_runs_or_fails_loud():
    """restormer's weights may or may not be present depending on the
    machine (this one has them downloaded - see test_restorers.py); either
    way build_panels() must not raise, and whichever branch is live must be
    internally consistent (a score without an error, or an error without a
    score, never both/neither)."""
    from src.demo.app import build_panels
    from src.models.deep_restorers import REPO_SPECS
    from src.utils.paths import checkpoints_dir

    panels = build_panels("illumination", 2)
    p = panels["restormer"]
    weights_present = (checkpoints_dir() / REPO_SPECS["restormer"]["weights"]).exists()
    if weights_present:
        assert p.error is None
        assert p.score is not None
    else:
        assert p.error is not None
        assert p.score is None
        assert np.array_equal(p.image, np.zeros_like(p.image))


@pytest.mark.parametrize("family", ["defocus", "motion", "illumination", "noise", "jpeg", "mixed"])
def test_build_panels_runs_for_every_family(family):
    from src.demo.app import build_panels
    build_panels(family, 3)  # must not raise


def test_overlay_heatmap_shape_and_range():
    from src.demo.app import overlay_heatmap
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    amap = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    out = overlay_heatmap(img, amap)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_overlay_heatmap_resizes_mismatched_map():
    from src.demo.app import overlay_heatmap
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    amap = np.random.default_rng(1).random((20, 20)).astype(np.float32)  # different shape
    out = overlay_heatmap(img, amap)
    assert out.shape == img.shape


def test_build_demo_constructs_blocks_without_launching():
    """Catches Gradio API breakage between the pinned >=4.0 and whatever's
    actually installed (6.28.0 here) without opening a server/port."""
    from src.demo.app import build_demo
    demo = build_demo()
    assert demo is not None


def test_panel_outputs_formats_error_and_score_text():
    from src.demo.app import _panel_outputs, PanelResult
    import numpy as np

    ok = PanelResult(image=np.zeros((4, 4, 3), np.float32), score=0.42, drr_rel=0.8)
    img, text = _panel_outputs(ok)
    assert "0.4200" in text and "0.800" in text

    bad = PanelResult(image=np.zeros((4, 4, 3), np.float32), score=None, error="no weights")
    _, text2 = _panel_outputs(bad)
    assert "UNAVAILABLE" in text2 and "no weights" in text2
