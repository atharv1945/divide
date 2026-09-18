import numpy as np
import pytest

from src.data.mvtec import synthetic_split
from src.detect.harness import AVAILABLE, DetectorHarness

pytest.importorskip("anomalib")
pytest.importorskip("lightning")


def _tiny_split():
    return synthetic_split("carpet", n_train=16, n_test_good=6, n_test_bad=6, size=48, seed=0)


@pytest.mark.parametrize("name", ["padim"])
def test_fit_and_score_fast_models(name):
    """PaDiM's fit is a single forward pass through a frozen backbone - fast
    enough to run on every CPU test invocation."""
    tr, te = _tiny_split()
    h = DetectorHarness(name, image_size=48, max_epochs=1)
    try:
        h.fit([s.image for s in tr])
        good = next(s for s in te if s.label == 0)
        bad = next(s for s in te if s.label == 1)
        r_good = h.score(good.image)
        r_bad = h.score(bad.image)
        assert np.isfinite(r_good.score)
        assert np.isfinite(r_bad.score)
        assert r_good.anomaly_map.shape == (48, 48)
        assert r_bad.anomaly_map.shape == (48, 48)
    finally:
        h.close()


def test_unknown_detector_name_raises():
    with pytest.raises(KeyError):
        DetectorHarness("not_a_real_detector")


def test_scoring_before_fit_raises():
    h = DetectorHarness("padim")
    try:
        with pytest.raises(RuntimeError, match="not been fit"):
            h.score(np.zeros((48, 48, 3), np.float32))
    finally:
        h.close()


def test_refitting_raises():
    tr, _ = _tiny_split()
    h = DetectorHarness("padim", image_size=48, max_epochs=1)
    try:
        h.fit([s.image for s in tr])
        with pytest.raises(RuntimeError, match="already fitted"):
            h.fit([s.image for s in tr])
    finally:
        h.close()


def test_available_names_are_exactly_the_four_frozen_detectors():
    assert set(AVAILABLE) == {"patchcore", "padim", "reverse_distillation", "efficientad"}
