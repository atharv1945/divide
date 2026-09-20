"""Metric unit tests against hand-constructed cases.

If these fail, every experimental result in the project is meaningless, so
these are the most important tests in the repo.
"""
import numpy as np
import pytest

from src.metrics.core import (
    auroc, average_precision, f1_max, psnr,
    defect_retention_ratio, defect_residual_correlation,
    anomaly_contrast_gain, local_contrast,
    degradation_magnitude, degradation_removal_ratio, hallucinated_defect_rate,
    robustness_gap, gap_closed,
)

RNG = np.random.default_rng(0)


def _base(h=64, w=64, val=0.5):
    return np.full((h, w, 3), val, np.float32)


def _mask(h=64, w=64):
    m = np.zeros((h, w), np.uint8)
    m[28:36, 20:44] = 1
    return m


# ---------------------------------------------------------------- DRR

def test_drr_perfect_preservation_is_one():
    """Restorer is the identity -> the defect survives untouched -> DRR == 1."""
    clean0 = _base()
    m = _mask()
    clean_a = clean0.copy()
    clean_a[m.astype(bool)] = 0.2                      # the defect

    # identity restoration
    drr = defect_retention_ratio(clean_a, clean0, clean_a, clean0, m)
    assert drr == pytest.approx(1.0, abs=1e-5)


def test_drr_total_erasure_is_zero():
    """Restorer outputs the same image whether or not the defect was there."""
    clean0 = _base()
    m = _mask()
    clean_a = clean0.copy()
    clean_a[m.astype(bool)] = 0.2

    restored_a = clean0.copy()      # defect wiped out
    restored_0 = clean0.copy()

    drr = defect_retention_ratio(restored_a, restored_0, clean_a, clean0, m)
    assert drr == pytest.approx(0.0, abs=1e-6)


def test_drr_half_preservation():
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    clean_a[mb] = 0.5 - 0.4                            # residual magnitude 0.4

    restored_a = clean0.copy()
    restored_a[mb] = 0.5 - 0.2                         # residual magnitude 0.2
    restored_0 = clean0.copy()

    drr = defect_retention_ratio(restored_a, restored_0, clean_a, clean0, m)
    assert drr == pytest.approx(0.5, abs=1e-5)


def test_drr_amplification_exceeds_one():
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    clean_a[mb] = 0.3

    restored_a = clean0.copy()
    restored_a[mb] = 0.1                               # residual doubled
    restored_0 = clean0.copy()

    drr = defect_retention_ratio(restored_a, restored_0, clean_a, clean0, m)
    assert drr > 1.0


def test_drr_empty_mask_is_nan():
    z = _base()
    m = np.zeros((64, 64), np.uint8)
    assert np.isnan(defect_retention_ratio(z, z, z, z, m))


def test_drr_only_looks_inside_mask():
    """Changes outside the mask must not affect DRR."""
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    clean_a[mb] = 0.2

    ra = clean_a.copy()
    r0 = clean0.copy()
    ra[0:5, 0:5] = 0.9                                 # noise far from defect
    r0[0:5, 0:5] = 0.9

    drr = defect_retention_ratio(ra, r0, clean_a, clean0, m)
    assert drr == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------- defect residual correlation

def test_residual_correlation_identity_is_one():
    """Perfect preservation: restored residual == true residual exactly ->
    correlation 1.0 (not just DRR 1.0 - the shape matches too, trivially,
    since it's the same signal)."""
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    clean_a[mb] += RNG.normal(0, 0.15, size=int(mb.sum()))[:, None]  # textured defect, not flat

    corr = defect_residual_correlation(clean_a, clean0, clean_a, clean0, m)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_residual_correlation_sign_flipped_is_minus_one():
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    defect = RNG.normal(0, 0.15, size=int(mb.sum()))
    clean_a[mb] += defect[:, None]

    restored_0 = clean0.copy()
    restored_a = clean0.copy()
    restored_a[mb] -= defect[:, None]  # exactly the negated residual

    corr = defect_residual_correlation(restored_a, restored_0, clean_a, clean0, m)
    assert corr == pytest.approx(-1.0, abs=1e-6)


def test_residual_correlation_near_zero_for_uncorrelated_noise_of_matched_magnitude():
    """The case this metric exists for: DRR would read this as roughly
    "preserved" (matched residual magnitude), but the restored residual's
    SHAPE has nothing to do with the true defect - independent random
    noise, not the defect's own signal."""
    clean0 = _base(h=128, w=128)
    m = np.zeros((128, 128), np.uint8)
    m[20:108, 20:108] = 1  # large mask for a stable correlation estimate
    mb = m.astype(bool)

    true_defect = RNG.normal(0, 0.15, size=int(mb.sum()))
    clean_a = clean0.copy()
    clean_a[mb] += true_defect[:, None]

    restored_0 = clean0.copy()
    restored_a = clean0.copy()
    ringing = RNG.normal(0, 0.15, size=int(mb.sum()))  # independent noise, matched sigma
    restored_a[mb] += ringing[:, None]

    drr = defect_retention_ratio(restored_a, restored_0, clean_a, clean0, m)
    corr = defect_residual_correlation(restored_a, restored_0, clean_a, clean0, m)
    assert drr == pytest.approx(1.0, rel=0.3)  # magnitude roughly matched - DRR alone looks fine
    assert abs(corr) < 0.3                     # but the shape is unrelated


def test_residual_correlation_empty_or_singleton_mask_is_nan():
    z = _base()
    assert np.isnan(defect_residual_correlation(z, z, z, z, np.zeros((64, 64), np.uint8)))
    single = np.zeros((64, 64), np.uint8)
    single[0, 0] = 1
    assert np.isnan(defect_residual_correlation(z, z, z, z, single))


def test_residual_correlation_only_looks_inside_mask():
    clean0 = _base()
    m = _mask()
    mb = m.astype(bool)
    clean_a = clean0.copy()
    defect = RNG.normal(0, 0.15, size=int(mb.sum()))
    clean_a[mb] += defect[:, None]

    ra = clean_a.copy()
    r0 = clean0.copy()
    ra[0:5, 0:5] = 0.9  # noise far from the defect, outside the mask
    r0[0:5, 0:5] = 0.2

    corr = defect_residual_correlation(ra, r0, clean_a, clean0, m)
    assert corr == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- ACG

def test_local_contrast_zero_when_defect_matches_background():
    img = _base()
    m = _mask()
    assert local_contrast(img, m) == pytest.approx(0.0, abs=1e-6)


def test_acg_greater_than_one_when_contrast_restored():
    m = _mask()
    mb = m.astype(bool)

    degraded = _base()
    degraded += RNG.normal(0, 0.02, degraded.shape).astype(np.float32)
    degraded[mb] -= 0.05                               # faint defect

    restored = _base()
    restored += RNG.normal(0, 0.02, restored.shape).astype(np.float32)
    restored[mb] -= 0.25                               # strong defect

    acg = anomaly_contrast_gain(restored, degraded, m)
    assert acg > 1.5


def test_acg_less_than_one_when_defect_smoothed():
    m = _mask()
    mb = m.astype(bool)

    degraded = _base()
    degraded += RNG.normal(0, 0.02, degraded.shape).astype(np.float32)
    degraded[mb] -= 0.30

    restored = _base()
    restored += RNG.normal(0, 0.02, restored.shape).astype(np.float32)
    restored[mb] -= 0.03                               # nearly erased

    assert anomaly_contrast_gain(restored, degraded, m) < 1.0


# ---------------------------------------------------------------- DRemR

def test_dremr_one_when_perfectly_restored():
    clean = _base()
    degraded = clean + 0.1
    restored = clean.copy()
    assert degradation_removal_ratio(restored, degraded, clean) == pytest.approx(1.0, abs=1e-6)


def test_dremr_zero_when_nothing_done():
    clean = _base()
    degraded = clean + 0.1
    assert degradation_removal_ratio(degraded, degraded, clean) == pytest.approx(0.0, abs=1e-6)


def test_dremr_negative_when_made_worse():
    clean = _base()
    degraded = clean + 0.1
    restored = clean + 0.3
    assert degradation_removal_ratio(restored, degraded, clean) < 0.0


def test_dremr_ignores_defect_region():
    clean = _base()
    m = _mask()
    mb = m.astype(bool)
    degraded = clean + 0.1
    restored = clean.copy()
    restored[mb] = 0.0                                 # huge error, but in-mask
    assert degradation_removal_ratio(restored, degraded, clean, m) == pytest.approx(1.0, abs=1e-6)


def test_degradation_magnitude_matches_dremr_denominator():
    """degradation_magnitude() must be exactly the quantity DRemR divides
    by - same fixed restoration error should produce a DRemR delta of
    error / degradation_magnitude, per the ratio's closed form."""
    clean = _base()
    degraded = clean + 0.1
    mag = degradation_magnitude(degraded, clean)
    assert mag == pytest.approx(0.1 * 64 * 64, rel=1e-5)  # _gray averages channels first

    restored_ok = clean.copy()
    restored_bad = clean + 0.02                        # small, fixed absolute error
    dremr_ok = degradation_removal_ratio(restored_ok, degraded, clean)
    dremr_bad = degradation_removal_ratio(restored_bad, degraded, clean)
    # grayscale (channel-averaged) error, matching _gray's convention inside
    # degradation_removal_ratio - NOT a raw per-channel RGB sum.
    error = float(np.abs(restored_bad - restored_ok).mean(axis=2).sum())
    assert (dremr_ok - dremr_bad) == pytest.approx(error / mag, rel=1e-4)


def test_degradation_magnitude_smaller_amplifies_dremr_delta_for_same_error():
    """The mechanism behind the ratio-instability hypothesis: identical
    absolute restoration error produces a LARGER DRemR swing when
    degradation_magnitude is smaller (mild degradation) than when it's
    larger (severe degradation)."""
    clean = _base()
    mild = clean + 0.02      # small degradation -> small denominator
    severe = clean + 0.5     # large degradation -> large denominator

    restored_ok_mild, restored_bad_mild = clean.copy(), clean + 0.01
    restored_ok_severe, restored_bad_severe = clean.copy(), clean + 0.01

    delta_mild = (degradation_removal_ratio(restored_ok_mild, mild, clean)
                 - degradation_removal_ratio(restored_bad_mild, mild, clean))
    delta_severe = (degradation_removal_ratio(restored_ok_severe, severe, clean)
                    - degradation_removal_ratio(restored_bad_severe, severe, clean))
    assert abs(delta_mild) > abs(delta_severe)


# ---------------------------------------------------------------- HDR

def test_hallucinated_defect_rate():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    assert hallucinated_defect_rate(scores, 0.5) == pytest.approx(0.5)
    assert hallucinated_defect_rate(scores, 0.99) == pytest.approx(0.0)


# ---------------------------------------------------------------- AUROC etc.

def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.0)


def test_auroc_ties_give_half():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.ones(4), y) == pytest.approx(0.5)


def test_auroc_known_value():
    # scores 1,2,3,4 with labels 0,1,0,1 -> AUROC = 0.75
    assert auroc(np.array([1., 2., 3., 4.]), np.array([0, 1, 0, 1])) == pytest.approx(0.75)


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.array([1., 2.]), np.array([0, 0])))


def test_average_precision_perfect():
    assert average_precision(np.array([0.1, 0.2, 0.8, 0.9]),
                             np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_f1_max_perfect_separation():
    f1, _ = f1_max(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert f1 == pytest.approx(1.0)


def test_psnr_identical_is_high():
    x = _base()
    assert psnr(x, x) > 90


def test_psnr_known_value():
    a = np.zeros((8, 8, 3), np.float32)
    b = np.full((8, 8, 3), 0.1, np.float32)
    assert psnr(a, b) == pytest.approx(20.0, abs=1e-4)


# ---------------------------------------------------------------- gap

def test_robustness_gap_and_gap_closed():
    assert robustness_gap(0.99, 0.80) == pytest.approx(0.19)
    assert gap_closed(0.99, 0.80, 0.895) == pytest.approx(0.5, abs=1e-6)
    assert gap_closed(0.99, 0.80, 0.80) == pytest.approx(0.0)
    assert gap_closed(0.99, 0.80, 0.99) == pytest.approx(1.0)


def test_relative_drr_normalises_against_identity():
    from src.metrics.core import relative_drr
    # identity itself always scores 1.0 relative to itself
    assert relative_drr(0.8, 0.8) == pytest.approx(1.0)
    # a restorer that halves what identity preserved
    assert relative_drr(0.4, 0.8) == pytest.approx(0.5)
    assert np.isnan(relative_drr(0.4, 0.0))
