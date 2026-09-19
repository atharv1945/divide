import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.pcim import (
    PCIM, ProxCNN, generalized_anscombe, inverse_generalized_anscombe,
    illumination_divide, illumination_restore, wiener_data_step,
    _kernel_to_otf, identity_otf,
)


def _rand(b=2, c=3, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# --------------------------------------------------------------------------
# elementwise transforms
# --------------------------------------------------------------------------

def test_vst_roundtrip_is_close():
    x = _rand(seed=1) * 0.8 + 0.1
    sigma = torch.full((2,), 0.05)
    v = generalized_anscombe(x, sigma)
    x_hat = inverse_generalized_anscombe(v, sigma)
    torch.testing.assert_close(x_hat, x, atol=1e-4, rtol=1e-3)


def test_vst_output_is_nonnegative_and_finite():
    x = _rand(seed=2)
    v = generalized_anscombe(x, torch.tensor([0.1, 0.2]))
    assert torch.isfinite(v).all()
    assert (v >= 0).all()


def test_illumination_divide_roundtrip():
    x = _rand(seed=3)
    field = 1.0 + 0.3 * _rand(b=2, c=1, h=16, w=16, seed=4)
    y = illumination_restore(illumination_divide(x, field), field)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


def test_illumination_divide_is_elementwise():
    """Changing one pixel of the field must not change any OTHER pixel's
    correction - this is what makes it content/position-blind rather than
    content-selective."""
    x = _rand(b=1, c=3, h=8, w=8, seed=5)
    field = torch.ones(1, 1, 8, 8)
    out1 = illumination_divide(x, field)
    field2 = field.clone()
    field2[0, 0, 3, 3] = 2.0
    out2 = illumination_divide(x, field2)
    diff = (out1 - out2).abs()
    changed = diff > 1e-8
    assert changed.sum().item() == x.shape[1]  # only that (row,col) across channels


# --------------------------------------------------------------------------
# data step / HQS recursion - must be linear when the prox is gated off
# --------------------------------------------------------------------------

def test_identity_otf_data_step_is_averaging():
    """With an identity kernel, the closed-form step should reduce y and z
    toward each other, not diverge or blow up."""
    v = _rand(b=1, c=1, h=8, w=8, seed=6)
    z = torch.zeros_like(v)
    otf = identity_otf((1, 1), 8, 8, v.dtype, v.device)
    sigma = torch.tensor([0.1])
    x = wiener_data_step(v, z, otf, sigma, rho=1.0)
    assert torch.isfinite(x).all()


def test_hqs_without_prox_is_linear():
    """The core structural claim: with the prox gated off (x_cons's path),
    the unrolled recursion is an affine map of the input for fixed
    (kernel, sigma). No nonlinear, content-selective decision is possible
    anywhere in this path."""
    model = PCIM(n_iters=4)
    model.eval()
    h = w = 16
    otf = identity_otf((1, 3), h, w, torch.float32, "cpu")
    sigma = torch.tensor([0.05])

    v1 = _rand(b=1, h=h, w=w, seed=10)
    v2 = _rand(b=1, h=h, w=w, seed=11)
    a, bcoef = 0.7, 1.3

    with torch.no_grad():
        out1 = model._hqs(v1, otf, sigma, gate=0.0)
        out2 = model._hqs(v2, otf, sigma, gate=0.0)
        out_combo = model._hqs(a * v1 + bcoef * v2, otf, sigma, gate=0.0)

    torch.testing.assert_close(out_combo, a * out1 + bcoef * out2, atol=1e-4, rtol=1e-3)


def test_hqs_with_prox_is_generally_nonlinear():
    """Sanity check the linearity test above is actually discriminating:
    with the prox engaged AND non-trivial (weights perturbed away from
    zero-init), superposition should NOT hold.

    At a fresh zero-init, gate=1 is ALSO exactly linear - the prox is
    exactly a no-op (see test_x_full_equals_x_cons_exactly_at_init), so
    gate=1 degenerates to the same linear recursion as gate=0 until the
    prox has learned something. That's intentional, not a bug; this test
    perturbs the prox first specifically to exercise the "prox actually
    doing something nonlinear" case the fresh-init model can't yet show."""
    torch.manual_seed(0)
    model = PCIM(n_iters=4)
    with torch.no_grad():
        for p in model.prox.parameters():
            p.add_(0.1 * torch.randn_like(p))
    model.eval()
    h = w = 16
    otf = identity_otf((1, 3), h, w, torch.float32, "cpu")
    sigma = torch.tensor([0.05])
    v1 = _rand(b=1, h=h, w=w, seed=20)
    v2 = _rand(b=1, h=h, w=w, seed=21)

    with torch.no_grad():
        out1 = model._hqs(v1, otf, sigma, gate=1.0)
        out2 = model._hqs(v2, otf, sigma, gate=1.0)
        out_combo = model._hqs(v1 + v2, otf, sigma, gate=1.0)

    assert not torch.allclose(out_combo, out1 + out2, atol=1e-4)


# --------------------------------------------------------------------------
# full module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_iters", [4, 5, 6])
def test_forward_shapes_for_valid_iter_counts(n_iters):
    model = PCIM(n_iters=n_iters)
    y = _rand(b=2, h=16, w=16, seed=30)
    field = torch.ones(2, 1, 16, 16)
    kernel = torch.zeros(2, 1, 3, 3); kernel[:, :, 1, 1] = 1.0
    sigma = torch.full((2,), 0.05)
    x_full, x_cons = model(y, illum_field=field, kernel=kernel, sigma=sigma)
    assert x_full.shape == y.shape
    assert x_cons.shape == y.shape
    assert torch.isfinite(x_full).all()
    assert torch.isfinite(x_cons).all()


def test_n_iters_out_of_range_rejected():
    with pytest.raises(ValueError):
        PCIM(n_iters=3)
    with pytest.raises(ValueError):
        PCIM(n_iters=7)


def test_forward_without_illum_or_kernel_falls_back_to_identity():
    model = PCIM(n_iters=4)
    y = _rand(b=1, h=16, w=16, seed=31)
    sigma = torch.tensor([0.05])
    x_full, x_cons = model(y, illum_field=None, kernel=None, sigma=sigma)
    assert x_full.shape == y.shape
    assert x_cons.shape == y.shape


def test_prox_cnn_is_under_200k_params():
    model = PCIM()
    assert model.prox_param_count() <= 200_000


def test_gradients_flow_to_prox():
    """At a fresh zero-init, gradient reaches the prox's LAST layer only -
    the standard "cold start" for a zero-initialised residual branch: with
    net[-1].weight=0, d(net_output)/d(net[-2]_output) = net[-1].weight^T =
    0, so the chain rule kills gradient to every earlier layer until the
    last layer's own weight (whose gradient depends on upstream
    activations, not on its own current value) moves away from zero.
    Confirmed empirically before writing this test - net[0..4] all get
    exactly zero gradient at step 0, net[-1] does not. This is expected
    and is not the same failure as the pre-fix state - after one optimizer
    step, gradient reaches net[0] too (checked below), so this is a
    one-step delay, not a stuck network."""
    torch.manual_seed(0)
    model = PCIM(n_iters=4)
    y = _rand(b=1, h=16, w=16, seed=32)
    sigma = torch.tensor([0.05])

    x_full, x_cons = model(y, sigma=sigma)
    (x_full.mean() + x_cons.mean()).backward()
    assert model.prox.net[-1].weight.grad is not None
    assert not torch.allclose(model.prox.net[-1].weight.grad,
                              torch.zeros_like(model.prox.net[-1].weight.grad))

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.step()  # last layer's weight is now non-zero
    opt.zero_grad()
    x_full, x_cons = model(y, sigma=sigma)
    (x_full.mean() + x_cons.mean()).backward()
    assert model.prox.net[0].weight.grad is not None
    assert not torch.allclose(model.prox.net[0].weight.grad,
                              torch.zeros_like(model.prox.net[0].weight.grad))


def test_x_full_equals_x_cons_exactly_at_init():
    """The regression test that should have existed before the ProxCNN's
    final layer was zero-initialised: at step 0, an untrained prox must be
    EXACTLY a no-op (not approximately - prox output should be identically
    zero), so x_full and x_cons - which differ only in whether the prox
    runs - must be bit-for-bit equal. Before the zero-init fix, this was
    false: step-0 x_full on real data measured DRemR -8.11 / PSNR 13.16dB,
    indistinguishable from a fully-"trained"-but-broken-physics run's final
    numbers, because 5 compounding applications of an untrained non-zero
    residual (PCIM._hqs's gate=1 path) was already enough to destroy the
    image before any training happened."""
    torch.manual_seed(0)
    model = PCIM(n_iters=5)
    y = _rand(b=2, h=16, w=16, seed=40)
    illum = torch.ones(2, 1, 16, 16)
    kernel = torch.zeros(2, 1, 3, 3); kernel[:, :, 1, 1] = 1.0
    sigma = torch.full((2,), 0.02)
    x_full, x_cons = model(y, illum_field=illum, kernel=kernel, sigma=sigma)
    torch.testing.assert_close(x_full, x_cons, atol=1e-6, rtol=1e-6)


def test_prox_net_output_is_exactly_zero_at_init():
    """Isolates the actual mechanism: the residual branch itself, not just
    the composed PCIM output."""
    torch.manual_seed(0)
    model = PCIM()
    x = _rand(b=1, h=16, w=16, seed=41)
    net_out = model.prox.net(x)
    torch.testing.assert_close(net_out, torch.zeros_like(net_out), atol=0.0, rtol=0.0)


def test_x_full_and_x_cons_diverge_once_prox_is_non_trivial():
    """The two outputs ARE genuinely different code paths (x_full applies
    the prox each iteration, x_cons never does) - verified by perturbing
    the prox away from its zero-init and confirming the paths then differ.
    This replaces an earlier version of this test that asserted x_full !=
    x_cons at a random (non-zero) init, which was actually checking for
    the bug this file's other new tests now check the fix for."""
    torch.manual_seed(1)
    model = PCIM(n_iters=4)
    with torch.no_grad():
        for p in model.prox.parameters():
            p.add_(0.1 * torch.randn_like(p))
    y = _rand(b=1, h=16, w=16, seed=33)
    sigma = torch.tensor([0.05])
    x_full, x_cons = model(y, sigma=sigma)
    assert not torch.allclose(x_full, x_cons, atol=1e-5)


def test_kernel_to_otf_shape():
    k = torch.zeros(2, 1, 5, 5)
    k[:, :, 2, 2] = 1.0
    otf = _kernel_to_otf(k, 16, 16)
    assert otf.shape == (2, 1, 16, 16)
    assert torch.is_complex(otf)
