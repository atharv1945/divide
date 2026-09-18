import numpy as np
import pytest

from src.experiments.dbde_validation import (
    defect_blindness_data, parameter_accuracy_data, reference_vs_blind_data,
    make_defect_blindness_figure, make_parameter_accuracy_figure,
    make_reference_vs_blind_figure, _pool_images,
)


def _images(n=6, size=96, seed=0):
    return _pool_images(["carpet", "bottle"], max(n // 2, 1), size, smoke=True, seed=seed)


def test_pool_images_returns_requested_count_and_shape():
    imgs = _images(n=4, size=64)
    assert len(imgs) >= 2
    for im in imgs:
        assert im.shape == (64, 64, 3)
        assert im.dtype == np.float32


def test_defect_blindness_data_is_flat_ish_and_finite():
    imgs = _images(n=6, size=80)
    data = defect_blindness_data(imgs, [0.01, 0.05], seed=0)
    assert set(data) == {"noise", "defocus", "motion"}
    for name, vals in data.items():
        assert len(vals) == 2
        assert all(np.isfinite(v) for v in vals)
        assert all(v >= 0 for v in vals)


def test_parameter_accuracy_data_has_five_severities_with_monotone_true_values():
    imgs = _images(n=4, size=80)
    data = parameter_accuracy_data(imgs, seed=0)
    assert set(data) == {"blur_radius", "blur_length", "noise_sigma"}
    for name, d in data.items():
        assert len(d["true"]) == 5
        assert d["true"] == sorted(d["true"])
        assert len(d["mean"]) == len(d["std"]) == 5


def test_reference_vs_blind_data_shape():
    imgs = _images(n=6, size=80)
    data = reference_vs_blind_data(imgs, seed=0)
    assert len(data["radii"]) == len(data["reference_mae"]) == len(data["blind_mae"]) == 5
    assert all(v >= 0 for v in data["reference_mae"])
    assert all(v >= 0 for v in data["blind_mae"])


def test_reference_vs_blind_raises_with_too_few_images():
    with pytest.raises(ValueError, match="held out"):
        reference_vs_blind_data(_images(n=2, size=64)[:2], seed=0)


def test_make_figures_write_files(tmp_path):
    imgs = _images(n=6, size=80)

    p1 = tmp_path / "blindness.png"
    d1 = make_defect_blindness_figure(imgs, p1, seed=0)
    assert p1.exists() and p1.stat().st_size > 0
    assert "noise" in d1

    p2 = tmp_path / "accuracy.png"
    d2 = make_parameter_accuracy_figure(imgs, p2, seed=0)
    assert p2.exists() and p2.stat().st_size > 0
    assert "blur_radius" in d2

    p3 = tmp_path / "ref_vs_blind.png"
    by_cat = {"carpet": imgs[:5], "bottle": imgs[5:] if len(imgs) > 5 else imgs[:4]}
    d3 = make_reference_vs_blind_figure(by_cat, p3, seed=0)
    assert p3.exists() and p3.stat().st_size > 0
    assert "relative_error_reduction" in d3


def test_reference_vs_blind_figure_requires_same_category_images_per_group():
    """A category with fewer than 4 images is skipped, not silently mixed
    with another category's images to make up the count."""
    imgs_a = _images(n=8, size=64)
    by_cat = {"carpet": imgs_a[:2], "bottle": imgs_a[2:8]}  # carpet has only 2
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        d = make_reference_vs_blind_figure(by_cat, Path(td) / "out.png", seed=0)
    assert "carpet" not in d["per_category"]
    assert "bottle" in d["per_category"]


def test_reference_vs_blind_figure_raises_when_no_category_qualifies():
    with pytest.raises(ValueError, match="no category"):
        make_reference_vs_blind_figure({"carpet": _images(n=2, size=64)[:2]}, "unused.png", seed=0)
