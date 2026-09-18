"""Tests for the degradation simulator and synthetic anomaly generator."""
import numpy as np
import pytest

from src.degrade.simulator import (
    FAMILIES, SEVERITIES, defocus_kernel, motion_kernel,
    sample_params, apply_degradation, degrade, degrade_pair,
    illumination_field, jpeg_compress,
)
from src.degrade.anomaly import (
    ANOMALY_KINDS, TextureBank, paste_anomaly, make_scratch, make_blob,
    perlin_mask, make_counterfactual_pair,
)

RNG = lambda s=0: np.random.default_rng(s)


def _img(h=96, w=96, seed=0):
    r = np.random.default_rng(seed)
    base = np.linspace(0.3, 0.7, w, dtype=np.float32)[None, :].repeat(h, 0)
    x = np.stack([base, base * 0.9, base * 1.05], axis=2)
    x += r.normal(0, 0.01, x.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- kernels

@pytest.mark.parametrize("r", [0.5, 1.0, 3.0, 6.0])
def test_defocus_kernel_normalised(r):
    k = defocus_kernel(r)
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    assert (k >= 0).all()


def test_defocus_kernel_grows_with_radius():
    assert defocus_kernel(6.0).shape[0] > defocus_kernel(1.0).shape[0]


@pytest.mark.parametrize("length,angle", [(3, 0.0), (10, np.pi / 4), (20, np.pi / 2)])
def test_motion_kernel_normalised(length, angle):
    k = motion_kernel(length, angle)
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    assert (k >= 0).all()


def test_motion_kernel_is_directional():
    """A horizontal kernel must spread along rows, not columns."""
    k = motion_kernel(11, 0.0)
    c = k.shape[0] // 2
    assert k[c, :].sum() > 5 * k[:, c].sum() - k[c, c]


# ---------------------------------------------------------------- params

@pytest.mark.parametrize("fam", FAMILIES)
@pytest.mark.parametrize("sev", SEVERITIES)
def test_sample_params_runs_for_every_cell(fam, sev):
    p = sample_params(fam, sev, RNG(1))
    assert p.family == fam and p.severity == sev
    v = p.vector()
    assert v.shape == (10,) and np.isfinite(v).all()


def test_sample_params_rejects_bad_input():
    with pytest.raises(ValueError):
        sample_params("nope", 1, RNG())
    with pytest.raises(ValueError):
        sample_params("noise", 9, RNG())


def test_severity_is_monotone_for_blur():
    radii = [sample_params("defocus", s, RNG(s)).blur_radius for s in SEVERITIES]
    assert radii == sorted(radii) and radii[0] < radii[-1]


def test_mixed_family_combines_multiple():
    """Mixed should activate at least two distinct degradation mechanisms."""
    hits = 0
    for seed in range(12):
        p = sample_params("mixed", 4, RNG(seed))
        active = sum([
            p.blur_kind != "none",
            p.illum_ev != 0.0 or p.illum_vignette != 0.0,
            p.noise_read > 0 or p.noise_shot > 0,
            p.jpeg_qf < 100,
        ])
        if active >= 2:
            hits += 1
    assert hits == 12


def test_mixed_never_stacks_both_blurs():
    for seed in range(20):
        p = sample_params("mixed", 3, RNG(seed))
        assert not (p.blur_radius > 0 and p.blur_length > 0)


# ---------------------------------------------------------------- application

@pytest.mark.parametrize("fam", FAMILIES)
def test_degrade_preserves_shape_and_range(fam):
    x = _img()
    y, p = degrade(x, fam, 3, seed=7)
    assert y.shape == x.shape and y.dtype == np.float32
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_degradation_is_deterministic_given_seed():
    x = _img()
    a, _ = degrade(x, "mixed", 4, seed=11)
    b, _ = degrade(x, "mixed", 4, seed=11)
    np.testing.assert_allclose(a, b)


def test_higher_severity_does_more_damage():
    x = _img()
    errs = [np.abs(degrade(x, "defocus", s, seed=3)[0] - x).mean() for s in SEVERITIES]
    assert errs[0] < errs[2] < errs[4]


def test_blur_reduces_high_frequency_energy():
    x = _img()
    y, _ = degrade(x, "defocus", 5, seed=2)
    hf = lambda im: np.abs(np.diff(im.mean(2), axis=1)).mean()
    assert hf(y) < hf(x)


def test_noise_increases_local_variance():
    x = _img()
    y, _ = degrade(x, "noise", 5, seed=2)
    resid = lambda im: np.std(np.diff(im.mean(2), axis=1))
    assert resid(y) > resid(x)


def test_illumination_changes_mean_brightness():
    x = _img()
    diffs = [abs(degrade(x, "illumination", 5, seed=s)[0].mean() - x.mean())
             for s in range(6)]
    assert max(diffs) > 0.05


def test_jpeg_lowers_quality_monotonically():
    x = _img()
    e_hi = np.abs(jpeg_compress(x, 90) - x).mean()
    e_lo = np.abs(jpeg_compress(x, 30) - x).mean()
    assert e_lo > e_hi


def test_illumination_field_is_smooth():
    """A low-order field must have negligible high-frequency content."""
    L = illumination_field((64, 64), 1.0, 0.3, RNG(0))
    assert np.abs(np.diff(L, axis=1)).max() < 0.05


# ---------------------------------------------------------------- pairs

def test_degrade_pair_shares_parameters_and_noise():
    """The counterfactual pair must differ ONLY where the images differ."""
    x0 = _img()
    xa = x0.copy()
    xa[40:48, 30:60] = 0.1

    ya, y0, p = degrade_pair(xa, x0, "noise", 4, seed=5)
    diff = np.abs(ya - y0).mean(axis=2)

    inside = diff[40:48, 30:60].mean()
    outside = np.delete(np.delete(diff, np.s_[40:48], 0), np.s_[30:60], 1).mean()
    assert inside > 10 * (outside + 1e-8)


def test_degrade_pair_identical_when_inputs_identical():
    x = _img()
    ya, y0, _ = degrade_pair(x, x, "mixed", 5, seed=9)
    np.testing.assert_allclose(ya, y0, atol=1e-6)


# ---------------------------------------------------------------- anomalies

@pytest.mark.parametrize("kind", ANOMALY_KINDS)
def test_paste_anomaly_shapes_and_mask(kind):
    x = _img()
    out, mask, spec = paste_anomaly(x, RNG(2), kind=kind)
    assert out.shape == x.shape
    assert mask.shape == x.shape[:2]
    assert set(np.unique(mask)).issubset({0, 1})
    assert spec.kind == kind


@pytest.mark.parametrize("kind", ANOMALY_KINDS)
def test_anomaly_is_sparse(kind):
    """Defects must stay small - that sparsity is the whole thesis."""
    x = _img(128, 128)
    fracs = []
    for s in range(6):
        _, mask, _ = paste_anomaly(x, RNG(s), kind=kind)
        fracs.append(mask.mean())
    assert 0.0 < np.mean(fracs) < 0.08


@pytest.mark.parametrize("kind", ANOMALY_KINDS)
def test_anomaly_changes_pixels_inside_mask(kind):
    x = _img()
    out, mask, _ = paste_anomaly(x, RNG(4), kind=kind)
    mb = mask.astype(bool)
    if mb.sum() == 0:
        pytest.skip("empty mask")
    assert np.abs(out - x).mean(axis=2)[mb].mean() > 0.01


@pytest.mark.parametrize("kind", ANOMALY_KINDS)
def test_anomaly_leaves_background_untouched(kind):
    x = _img()
    out, mask, _ = paste_anomaly(x, RNG(4), kind=kind)
    far = ~(mask.astype(bool))
    # dilate-free check: most background pixels should be identical
    unchanged = (np.abs(out - x).mean(axis=2)[far] < 1e-3).mean()
    assert unchanged > 0.85


def test_scratch_is_thin():
    """A scratch must be a thin structure - this is the hardest case."""
    m = make_scratch((128, 128), RNG(3))
    assert 0 < m.mean() < 0.05


def test_blob_is_connected_and_smooth():
    m = make_blob((128, 128), RNG(3), target_frac=0.01)
    assert m.max() > 0.5
    assert np.abs(np.diff(m, axis=1)).max() < 0.5


def test_perlin_mask_hits_target_area():
    m = perlin_mask((128, 128), RNG(1), target_frac=0.03)
    assert 0.002 < m.mean() < 0.12


def test_texture_bank_falls_back_without_dtd():
    bank = TextureBank(None)
    assert not bank.available
    t = bank.sample((32, 32), RNG(0))
    assert t.shape == (32, 32, 3) and 0 <= t.min() and t.max() <= 1


def test_counterfactual_pair_helper():
    x = _img()
    xa, x0, mask, spec = make_counterfactual_pair(x, RNG(6))
    np.testing.assert_allclose(x0, x)
    assert mask.sum() > 0
    assert np.abs(xa - x0).sum() > 0


def test_anomaly_deterministic_given_seed():
    x = _img()
    a, ma, _ = paste_anomaly(x, RNG(42))
    b, mb, _ = paste_anomaly(x, RNG(42))
    np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(ma, mb)
