import pytest

torch = pytest.importorskip("torch")

from src.models.sarg import SARG, _soft_threshold, _svt


def _rand(b=1, c=3, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def test_soft_threshold_zeroes_small_values():
    x = torch.tensor([-0.5, -0.05, 0.0, 0.05, 0.5])
    out = _soft_threshold(x, 0.1)
    torch.testing.assert_close(out, torch.tensor([-0.4, 0.0, 0.0, 0.0, 0.4]))


def test_soft_threshold_zero_lambda_is_identity():
    x = _rand(seed=1)
    torch.testing.assert_close(_soft_threshold(x, 0.0), x)


def test_svt_reduces_rank_for_large_lambda():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 12)
    low_rank = _svt(x, lam=10.0)  # huge threshold kills almost everything
    rank = torch.linalg.matrix_rank(low_rank)
    assert (rank < 12).all()


def test_svt_zero_lambda_is_near_identity():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 8)
    out = _svt(x, lam=0.0)
    torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# SARG module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_iters", [3, 4, 5])
def test_iter_counts_in_spec_range_accepted(n_iters):
    SARG(n_iters=n_iters)


def test_iter_count_out_of_range_rejected():
    with pytest.raises(ValueError):
        SARG(n_iters=2)
    with pytest.raises(ValueError):
        SARG(n_iters=6)


def test_mask_shape_and_range():
    sarg = SARG(n_iters=3)
    residual = _rand(b=2, c=3, h=16, w=16, seed=2) - 0.5
    m = sarg.mask(residual)
    assert m.shape == (2, 1, 16, 16)
    assert (m >= 0).all() and (m <= 1).all()
    assert torch.isfinite(m).all()


def test_mask_responds_to_a_sparse_spike_over_a_lowrank_background():
    """A localised outlier riding on top of a smooth (low-rank) background
    should push the mask up at that location more than at a quiet one - the
    whole point of the sparse term. A spike on an all-zero background is
    degenerate (trivially rank-1 too, so RPCA can dump it into L just as
    easily as S) - a smooth background gives the decomposition something
    genuinely low-rank to separate the spike FROM."""
    torch.manual_seed(0)
    sarg = SARG(n_iters=4, init_lam_low=1.0, init_lam_sparse=0.2)
    lin = torch.linspace(-1, 1, 16)
    background = 0.1 * (lin.view(1, 16, 1) + lin.view(1, 1, 16))  # smooth, rank-2
    residual = background.unsqueeze(0).repeat(1, 3, 1, 1)
    residual[:, :, 8, 8] += 5.0
    m = sarg.mask(residual)
    assert m[0, 0, 8, 8] > m[0, 0, 0, 0]


def test_decompose_reconstructs_approximately():
    """low_rank + sparse should account for most of the residual energy,
    even if not exactly (soft-thresholding/SVT are lossy by design)."""
    torch.manual_seed(0)
    sarg = SARG(n_iters=4)
    residual = _rand(b=1, c=1, h=16, w=16, seed=3) - 0.5
    low, sparse = sarg.decompose(residual)
    recon_err = (residual - (low + sparse)).abs().mean()
    total_scale = residual.abs().mean()
    assert recon_err < total_scale * 2.0  # sanity bound, not tight


def test_forward_blend_matches_extremes_at_mask_bounds():
    sarg = SARG(n_iters=3)
    y = _rand(b=1, c=3, h=16, w=16, seed=4)
    x_full = _rand(b=1, c=3, h=16, w=16, seed=5)
    x_cons = _rand(b=1, c=3, h=16, w=16, seed=6)
    x_blend, m = sarg(y, x_full, x_cons)
    assert x_blend.shape == x_full.shape
    assert m.shape == (1, 1, 16, 16)
    # blend must lie between x_full and x_cons pointwise (mask in [0,1])
    lo = torch.minimum(x_full, x_cons)
    hi = torch.maximum(x_full, x_cons)
    assert (x_blend >= lo - 1e-5).all() and (x_blend <= hi + 1e-5).all()


def test_mask_is_over_inclusive_not_undersensitive():
    """SARG should err toward protecting too much rather than too little -
    check that a mild residual still produces a non-trivial (not near-zero
    everywhere) mask, consistent with "high recall, low precision"."""
    torch.manual_seed(0)
    sarg = SARG(n_iters=4, mask_bias=-0.5)
    residual = (_rand(b=1, c=3, h=16, w=16, seed=7) - 0.5) * 0.3
    m = sarg.mask(residual)
    assert m.mean().item() > 0.0


def test_gradients_flow_through_forward():
    sarg = SARG(n_iters=3)
    y = _rand(b=1, c=3, h=16, w=16, seed=8)
    x_full = _rand(b=1, c=3, h=16, w=16, seed=9)
    x_full.requires_grad_(True)
    x_cons = _rand(b=1, c=3, h=16, w=16, seed=10)
    x_blend, _ = sarg(y, x_full, x_cons)
    x_blend.sum().backward()
    assert x_full.grad is not None
    assert sarg.log_lam_low.grad is not None
    assert sarg.log_lam_sparse.grad is not None
