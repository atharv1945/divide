import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.losses import (
    reconstruction_loss, degradation_consistency_loss, frequency_loss,
    preservation_loss, divide_loss,
)
from src.models.pcim import identity_otf


def _rand(b=1, c=3, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# --------------------------------------------------------------------------
# L_rec
# --------------------------------------------------------------------------

def test_reconstruction_loss_zero_when_identical():
    x = _rand(seed=1)
    assert reconstruction_loss(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_reconstruction_loss_positive_when_different():
    x = _rand(seed=2)
    y = _rand(seed=3)
    assert reconstruction_loss(x, y).item() > 0.0


# --------------------------------------------------------------------------
# L_deg
# --------------------------------------------------------------------------

def test_degradation_consistency_zero_when_xhat_equals_observed_no_ops():
    x_hat = _rand(seed=4)
    loss = degradation_consistency_loss(x_hat, x_hat, illum_field=None, kernel=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_degradation_consistency_reflects_illumination_mismatch():
    x_hat = torch.full((1, 3, 8, 8), 0.5)
    field = torch.full((1, 1, 8, 8), 2.0)
    y_observed = x_hat  # observed does NOT reflect the 2x field
    loss_with_field = degradation_consistency_loss(x_hat, y_observed, illum_field=field)
    loss_without_field = degradation_consistency_loss(x_hat, y_observed, illum_field=None)
    assert loss_with_field.item() > loss_without_field.item()


def test_degradation_consistency_zero_when_observed_matches_reapplied_field():
    x_hat = torch.full((1, 3, 8, 8), 0.5)
    field = torch.full((1, 1, 8, 8), 2.0)
    y_observed = x_hat * 2.0
    loss = degradation_consistency_loss(x_hat, y_observed, illum_field=field)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------
# L_freq
# --------------------------------------------------------------------------

def test_frequency_loss_zero_when_identical():
    x = _rand(seed=5)
    assert frequency_loss(x, x).item() == pytest.approx(0.0, abs=1e-4)


def test_frequency_loss_positive_when_different():
    x = _rand(seed=6)
    y = _rand(seed=7)
    assert frequency_loss(x, y).item() > 0.0


# --------------------------------------------------------------------------
# L_pres - the central novelty
# --------------------------------------------------------------------------

def test_preservation_loss_zero_when_restorer_preserves_the_residual_exactly():
    x_a = _rand(b=1, c=1, h=8, w=8, seed=8)
    x_0 = _rand(b=1, c=1, h=8, w=8, seed=9)
    mask = torch.zeros(1, 8, 8); mask[0, 2:5, 2:5] = 1.0
    # restorer does nothing at all - residual is trivially preserved
    x_tilde_a, x_tilde_0 = x_a.clone(), x_0.clone()
    loss = preservation_loss(x_tilde_a, x_tilde_0, x_a, x_0, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_preservation_loss_positive_when_restorer_erases_the_defect():
    x_a = torch.zeros(1, 1, 8, 8)
    x_a[0, 0, 3, 3] = 1.0  # a single "defect" pixel
    x_0 = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros(1, 8, 8); mask[0, 3, 3] = 1.0

    # restorer erases the defect: x_tilde_a == x_tilde_0
    x_tilde_a = x_0.clone()
    x_tilde_0 = x_0.clone()
    loss = preservation_loss(x_tilde_a, x_tilde_0, x_a, x_0, mask)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)  # full residual lost


def test_preservation_loss_ignores_changes_outside_mask():
    x_a = _rand(b=1, c=1, h=8, w=8, seed=10)
    x_0 = _rand(b=1, c=1, h=8, w=8, seed=11)
    mask = torch.zeros(1, 8, 8); mask[0, 0, 0] = 1.0  # tiny mask, one pixel

    x_tilde_a = x_a.clone()
    x_tilde_0 = x_0.clone()
    base = preservation_loss(x_tilde_a, x_tilde_0, x_a, x_0, mask)

    # perturb the restorer output far outside the mask
    x_tilde_a2 = x_tilde_a.clone()
    x_tilde_a2[0, 0, 6, 6] += 5.0
    perturbed = preservation_loss(x_tilde_a2, x_tilde_0, x_a, x_0, mask)

    assert base.item() == pytest.approx(perturbed.item(), abs=1e-6)


def test_preservation_loss_accepts_both_3d_and_4d_masks():
    x_a = _rand(b=1, c=1, h=8, w=8, seed=12)
    x_0 = _rand(b=1, c=1, h=8, w=8, seed=13)
    mask3 = torch.zeros(1, 8, 8); mask3[0, 1, 1] = 1.0
    mask4 = mask3.unsqueeze(1)
    l3 = preservation_loss(x_a, x_0, x_a, x_0, mask3)
    l4 = preservation_loss(x_a, x_0, x_a, x_0, mask4)
    torch.testing.assert_close(l3, l4)


def test_preservation_loss_empty_mask_does_not_nan_or_crash():
    x_a = _rand(b=1, c=1, h=8, w=8, seed=14)
    x_0 = _rand(b=1, c=1, h=8, w=8, seed=15)
    mask = torch.zeros(1, 8, 8)
    loss = preservation_loss(x_a, x_0, x_a, x_0, mask)
    assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------
# counterfactual pair integration - the fragile wiring L_pres depends on
# --------------------------------------------------------------------------

def test_preservation_loss_with_real_counterfactual_pair():
    """End-to-end sanity check using the actual degrade_pair()/paste_anomaly()
    pipeline, not hand-built tensors - this is the wiring the README calls
    out as fragile and invisible when broken."""
    from src.degrade.simulator import degrade_pair
    from src.degrade.anomaly import paste_anomaly

    rng = np.random.default_rng(0)
    clean0 = rng.random((16, 16, 3)).astype(np.float32) * 0.6 + 0.2
    clean_a, mask_np, _ = paste_anomaly(clean0, rng, kind="blob")
    assert mask_np.sum() > 0, "test fixture produced an empty mask - regenerate"

    y_a, y_0, _ = degrade_pair(clean_a, clean0, "noise", severity=3, seed=1)

    to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float()
    x_a, x_0 = to_t(clean_a), to_t(clean0)
    x_tilde_a, x_tilde_0 = to_t(y_a), to_t(y_0)  # "restorer" = identity, for this check
    mask = torch.from_numpy(mask_np).unsqueeze(0).float()

    loss = preservation_loss(x_tilde_a, x_tilde_0, x_a, x_0, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


# --------------------------------------------------------------------------
# combined loss + gradients
# --------------------------------------------------------------------------

def test_divide_loss_returns_all_components_and_gradients_flow():
    x_full = _rand(seed=20, c=1)
    x_full.requires_grad_(True)
    x_clean = _rand(seed=21, c=1)
    y_observed = _rand(seed=22, c=1)
    x_tilde_a = _rand(seed=23, c=1)
    x_tilde_0 = _rand(seed=24, c=1)
    x_a = _rand(seed=25, c=1)
    x_0 = _rand(seed=26, c=1)
    mask = torch.zeros(1, 16, 16); mask[0, 4:8, 4:8] = 1.0

    out = divide_loss(x_full, x_clean, y_observed, x_tilde_a, x_tilde_0, x_a, x_0, mask)
    assert set(out) == {"total", "rec", "deg", "freq", "pres"}
    for v in out.values():
        assert torch.isfinite(v)
    out["total"].backward()
    assert x_full.grad is not None
    assert not torch.allclose(x_full.grad, torch.zeros_like(x_full.grad))
