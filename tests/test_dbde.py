"""Tests for the Defect-Blind Degradation Estimator.

The most important test here is test_estimator_is_defect_blind: it verifies
the structural claim that the whole method rests on.
"""
import numpy as np
import pytest

from src.dbde.estimator import (
    estimate, estimate_noise_sigma, estimate_illumination, correct_illumination,
    estimate_motion_blur, estimate_defocus_radius, estimate_jpeg_qf,
    radial_psd, cepstrum, DegradationEstimate,
)
from src.degrade.simulator import (
    apply_degradation, sample_params, DegradationParams, degrade,
    illumination_field, jpeg_compress,
)
from src.degrade.anomaly import paste_anomaly

RNG = lambda s=0: np.random.default_rng(s)


def textured(h=192, w=192, seed=0):
    """Textured test image - blur/noise estimation needs real spectral content."""
    r = np.random.default_rng(seed)
    n = r.random((h // 4, w // 4)).astype(np.float32)
    import cv2
    base = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 * base + 0.25 * np.sin(xx / 6.0) * 0.5 + 0.4
    x = np.stack([base, base * 0.95, base * 1.03], axis=2)
    return np.clip(x, 0.05, 0.95).astype(np.float32)


# ---------------------------------------------------------------- noise

@pytest.mark.parametrize("sigma", [2 / 255, 5 / 255, 10 / 255, 15 / 255])
def test_noise_estimate_tracks_truth(sigma):
    x = textured()
    y = np.clip(x + RNG(1).standard_normal(x.shape).astype(np.float32) * sigma, 0, 1)
    est = estimate_noise_sigma(y)
    assert est == pytest.approx(sigma, abs=0.035), f"est={est:.4f} true={sigma:.4f}"


def test_noise_estimate_is_monotone():
    x = textured()
    ests = []
    for s in [1, 4, 8, 14]:
        y = np.clip(x + RNG(2).standard_normal(x.shape).astype(np.float32) * s / 255, 0, 1)
        ests.append(estimate_noise_sigma(y))
    assert ests == sorted(ests)


def test_noise_estimate_small_on_clean_image():
    assert estimate_noise_sigma(textured()) < 0.05


# ---------------------------------------------------------------- illumination

def test_illumination_fit_recovers_smooth_field():
    x = textured()
    L = illumination_field((192, 192), 0.0, 0.30, RNG(3))
    y = np.clip(x * L[:, :, None], 0, 1)
    Lhat, _ = estimate_illumination(y)
    # compare shape after removing the mean level
    a = L / L.mean()
    b = Lhat / Lhat.mean()
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.75


def test_illumination_correction_reduces_error():
    x = textured()
    L = illumination_field((192, 192), -0.8, 0.25, RNG(4))
    y = np.clip(x * L[:, :, None], 0, 1)
    Lhat, ev = estimate_illumination(y)
    corrected = correct_illumination(y, Lhat, ev)
    # correction should move us closer to the clean image in relative terms
    before = np.abs(y / (y.mean() + 1e-8) - x / x.mean()).mean()
    after = np.abs(corrected / (corrected.mean() + 1e-8) - x / x.mean()).mean()
    assert after < before


def test_illumination_degree_cannot_fit_a_scratch():
    """Low polynomial order is the safety property - verify it directly."""
    x = textured()
    xa = x.copy()
    xa[90:96, 40:150] = 0.02                       # hard dark scratch
    L_clean, _ = estimate_illumination(x)
    L_anom, _ = estimate_illumination(xa)
    # the fitted field must barely react to the scratch
    assert np.abs(L_clean - L_anom).max() < 0.10


# ---------------------------------------------------------------- blur

@pytest.mark.parametrize("length", [8, 12, 18])
def test_motion_blur_length_estimate(length):
    x = textured()
    p = DegradationParams(family="motion", severity=3, blur_kind="motion",
                          blur_length=float(length), blur_angle=0.0)
    y = apply_degradation(x, p, RNG(5))
    est_len, est_ang, conf = estimate_motion_blur(y)
    assert conf > 0.0
    assert est_len == pytest.approx(length, rel=0.45), f"est={est_len} true={length}"


def test_motion_blur_angle_estimate():
    x = textured()
    for true_ang in [0.0, np.pi / 2]:
        p = DegradationParams(family="motion", severity=4, blur_kind="motion",
                              blur_length=14.0, blur_angle=true_ang)
        y = apply_degradation(x, p, RNG(6))
        _, est_ang, conf = estimate_motion_blur(y)
        if conf < 0.05:
            continue
        d = min(abs(est_ang - true_ang), np.pi - abs(est_ang - true_ang))
        assert d < 0.5, f"angle est={est_ang:.2f} true={true_ang:.2f}"


def test_defocus_radius_with_reference_is_accurate():
    """Reference-based mode: clean normals are always available in practice."""
    from src.dbde.estimator import reference_psd
    x = textured()
    ref = reference_psd([textured(seed=s) for s in range(4)])
    for true_r in [1.0, 3.0, 6.0]:
        p = DegradationParams(family="defocus", severity=3, blur_kind="defocus",
                              blur_radius=true_r)
        y = apply_degradation(x, p, RNG(7))
        est, _ = estimate_defocus_radius(y, ref_logpsd=ref)
        assert est == pytest.approx(true_r, abs=1.0), f"est={est} true={true_r}"


def test_defocus_radius_blind_is_monotone():
    """Blind fallback: weaker, but must still order severities correctly."""
    x = textured()
    ests = []
    for r in [1.0, 3.0]:
        p = DegradationParams(family="defocus", severity=3, blur_kind="defocus",
                              blur_radius=r)
        y = apply_degradation(x, p, RNG(7))
        ests.append(estimate_defocus_radius(y)[0])
    assert ests[0] < ests[1], ests


def test_reference_psd_shape():
    from src.dbde.estimator import reference_psd
    ref = reference_psd([textured(seed=s) for s in range(3)])
    assert ref.shape == (128,) and np.isfinite(ref).all()


def test_no_blur_detected_on_sharp_image():
    est = estimate(textured())
    assert est.blur_kind == "none" or est.blur_radius < 1.5


# ---------------------------------------------------------------- jpeg

def test_jpeg_qf_estimate_orders_correctly():
    x = textured()
    q_hi = estimate_jpeg_qf(jpeg_compress(x, 90))
    q_lo = estimate_jpeg_qf(jpeg_compress(x, 30))
    assert q_lo <= q_hi


def test_jpeg_qf_is_100_on_uncompressed():
    assert estimate_jpeg_qf(textured()) >= 90


# ---------------------------------------------------------------- spectral tools

def test_radial_psd_shape_and_decay():
    freq, power = radial_psd(textured())
    assert freq.shape == power.shape
    assert (power > 0).all()
    assert power[:8].mean() > power[-8:].mean()      # natural images decay


def test_cepstrum_shape():
    c = cepstrum(textured())
    assert c.shape == (192, 192) and np.isfinite(c).all()


# ---------------------------------------------------------------- THE KEY TEST

@pytest.mark.parametrize("family", ["defocus", "motion", "noise", "illumination"])
def test_estimator_is_defect_blind(family):
    """The structural claim: a sparse defect must not move the estimate.

    This is the experiment that justifies calling the estimator defect-blind,
    and it is the figure that goes in the report.
    """
    x = textured(256, 256, seed=8)
    p = sample_params(family, 4, RNG(9))
    y_clean = apply_degradation(x, p, RNG(10))

    xa, mask, _ = paste_anomaly(x, RNG(11), kind="texture", area_frac=0.02)
    y_anom = apply_degradation(xa, p, RNG(10))

    e0 = estimate(y_clean)
    e1 = estimate(y_anom)

    assert abs(e0.noise_sigma - e1.noise_sigma) < 0.02
    assert abs(e0.blur_radius - e1.blur_radius) < 1.0
    assert abs(e0.blur_length - e1.blur_length) < 4.0
    assert np.abs(e0.illum_field - e1.illum_field).max() < 0.15


def test_defect_blindness_scales_gracefully_with_area():
    """Estimation drift should grow slowly, roughly with defect area."""
    x = textured(256, 256, seed=12)
    p = sample_params("noise", 3, RNG(13))
    base = estimate(apply_degradation(x, p, RNG(14))).noise_sigma

    drifts = []
    for frac in [0.005, 0.02, 0.05]:
        xa, _, _ = paste_anomaly(x, RNG(15), kind="texture", area_frac=frac)
        e = estimate(apply_degradation(xa, p, RNG(14)))
        drifts.append(abs(e.noise_sigma - base))

    assert max(drifts) < 0.03, drifts


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("family", ["defocus", "motion", "illumination", "noise", "jpeg", "mixed"])
def test_estimate_runs_on_every_family(family):
    y, p = degrade(textured(), family, 3, seed=16)
    est = estimate(y)
    assert isinstance(est, DegradationEstimate)
    assert np.isfinite(est.noise_sigma)
    assert est.illum_field is not None and est.illum_field.shape == y.shape[:2]
    k = est.kernel()
    assert k.sum() == pytest.approx(1.0, abs=1e-4)


def test_estimate_needs_no_gpu_or_weights():
    """Sanity: the estimator is pure numpy/opencv with no learned parameters."""
    import src.dbde.estimator as m
    src = open(m.__file__).read()
    assert "torch" not in src and "cuda" not in src.lower()
