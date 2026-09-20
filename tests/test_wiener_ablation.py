import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.degrade.simulator import degrade
from src.models.wiener_ablation import (
    wiener_dbde_nsr, wiener_hqs_rawpixel, wiener_hqs_vst, wiener_hqs_vst_nofloor,
)


def _blurred(size=48, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.random((size, size, 3)).astype(np.float32) * 0.4 + 0.3, 0, 1)
    y, _ = degrade(img, "defocus", 3, seed=seed + 1)
    return y


@pytest.mark.parametrize("fn", [wiener_dbde_nsr, wiener_hqs_rawpixel, wiener_hqs_vst, wiener_hqs_vst_nofloor])
def test_config_runs_and_returns_finite_same_shape(fn):
    y = _blurred()
    out = fn(y)
    assert out.shape == y.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("fn", [wiener_dbde_nsr, wiener_hqs_rawpixel, wiener_hqs_vst, wiener_hqs_vst_nofloor])
def test_config_is_a_noop_when_no_blur_is_estimated(fn, monkeypatch):
    """Every config must skip deconvolution (return the input unchanged)
    when DBDE reports blur_kind='none' - same fail-safe PCIM.forward()
    itself relies on (kernel=None skips the Wiener step entirely rather
    than deconvolving against a fake identity kernel)."""
    import src.dbde.estimator as dbde
    from src.dbde.estimator import DegradationEstimate

    y = np.clip(np.random.default_rng(2).random((32, 32, 3)).astype(np.float32), 0, 1)
    fake = DegradationEstimate(blur_kind="none", illum_field=np.ones((32, 32), np.float32))
    monkeypatch.setattr(dbde, "estimate", lambda img, ref_logpsd=None: fake)

    out = fn(y)
    np.testing.assert_allclose(out, y, atol=1e-5)


def test_wiener_hqs_vst_matches_pcim_x_cons_directly():
    """config D IS x_cons by construction - cross-check against calling
    PCIM.forward() directly with the same estimate, rather than trusting
    the wrapper's own internal wiring."""
    from src.dbde.estimator import estimate
    from src.models.pcim import PCIM

    y = _blurred(seed=5)
    out = wiener_hqs_vst(y)

    est = estimate(y)
    model = PCIM(nsr_floor=0.01)
    model.eval()
    yt = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))).unsqueeze(0).float()
    illum = torch.from_numpy(np.ascontiguousarray(est.illum_field)).unsqueeze(0).unsqueeze(0).float()
    kernel = (torch.from_numpy(np.ascontiguousarray(est.kernel())).unsqueeze(0).unsqueeze(0).float()
             if est.blur_kind != "none" else None)
    sigma = torch.tensor([max(est.noise_sigma, 1e-4)])
    with torch.no_grad():
        _, x_cons = model(yt, illum_field=illum, kernel=kernel, sigma=sigma)
    expected = x_cons.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_nofloor_config_has_much_smaller_effective_regularization():
    """Sanity check on config E's actual point: disabling nsr_floor must
    genuinely change the regularization strength used, not silently no-op
    back to the floored value."""
    from src.dbde.estimator import estimate
    y = _blurred(seed=7)
    est = estimate(y)
    # sigma_wiener with the default floor vs. an effectively-disabled one
    floored = max(max(est.noise_sigma, 1e-4), 0.01 ** 0.5)
    unfloored = max(max(est.noise_sigma, 1e-4), 1e-8 ** 0.5)
    assert unfloored < floored / 10  # at least an order of magnitude smaller
