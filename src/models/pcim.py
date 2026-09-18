"""PCIM - Physics-Consistent Inverse Module.

Inverts the physical degradation chain DBDE estimated (illumination field,
blur kernel, noise level), rather than learning a general denoiser. Structure:

    y --[illum divide]--> --[generalised Anscombe VST]--> v
    v --[N unrolled half-quadratic-splitting iterations]--> v~
    v~ --[inverse VST]--> x~

Each HQS iteration has two steps:
    data step   - closed-form FFT Wiener solve. Depends only on the estimated
                  kernel/noise and a FIXED (not learned) rho schedule. No
                  parameters, so nothing here can treat one pixel differently
                  from another based on what's IN the image.
    prox step   - a tiny learned CNN (<=200K params), gated by a scalar alpha.

Two outputs, both produced by the SAME recursion with the prox step gated
differently:
    x_full  (alpha=1, prox fully engaged)   - best reconstruction quality,
                                               but the CNN COULD learn to
                                               erase small anomalies, same as
                                               any denoiser.
    x_cons  (alpha=0, prox gated off)       - the data step alone, iterated.
                                               This composition is an affine
                                               function of the input for fixed
                                               (kernel, sigma) - see
                                               test_pcim.py::test_hqs_without_prox_is_linear.
                                               No nonlinear network ever sees
                                               these pixels, so it structurally
                                               cannot make a content-based
                                               decision to remove them. This is
                                               the "structural, not learned"
                                               half of the DIVIDE claim.

SARG (not built yet) blends between the two per-pixel using a protection mask.

`alpha` is also exposed as a learned nn.Parameter (sigmoid-squashed to
[0, 1]) for training a single deployable output between the two extremes;
`forward()` always returns the x_full / x_cons pair regardless, since L_pres
and SARG both need both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------
# elementwise, content-agnostic by construction (no cross-pixel mixing)
# --------------------------------------------------------------------------

def illumination_divide(y: torch.Tensor, illum_field: torch.Tensor,
                        clamp: float = 4.0) -> torch.Tensor:
    """Divide out an already-ESTIMATED illumination field (from DBDE).

    illum_field: (B,1,H,W) or (B,H,W), multiplicative, unit geometric mean.
    Purely elementwise - a pixel's correction depends only on its own
    position in the fixed field, never on neighbouring pixel VALUES.
    """
    if illum_field.dim() == 3:
        illum_field = illum_field.unsqueeze(1)
    field = illum_field.clamp(1.0 / clamp, clamp)
    return y / field


def illumination_restore(x: torch.Tensor, illum_field: torch.Tensor,
                         clamp: float = 4.0) -> torch.Tensor:
    """Inverse of illumination_divide - re-applies the field."""
    if illum_field.dim() == 3:
        illum_field = illum_field.unsqueeze(1)
    field = illum_field.clamp(1.0 / clamp, clamp)
    return x * field


def generalized_anscombe(x: torch.Tensor, sigma: torch.Tensor,
                         gain: float = 1.0) -> torch.Tensor:
    """Variance-stabilising transform for mixed Poisson-Gaussian noise.

        v = (2/gain) * sqrt(gain*x + 3/8*gain^2 + sigma^2)

    `sigma` is the DBDE-estimated noise level (per-image scalar or per-pixel
    map, broadcastable to x). `gain` is a fixed nominal photon gain - DBDE's
    blind estimator reports one combined noise sigma rather than separately
    decomposed shot/read components, so this VST is "generalised" in form
    but calibrated with a single estimated parameter rather than two. Purely
    elementwise.
    """
    sigma = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    while sigma.dim() < x.dim():
        sigma = sigma.unsqueeze(-1)
    inner = (gain * x + 0.375 * gain ** 2 + sigma ** 2).clamp_min(EPS)
    return (2.0 / gain) * torch.sqrt(inner)


def inverse_generalized_anscombe(v: torch.Tensor, sigma: torch.Tensor,
                                 gain: float = 1.0) -> torch.Tensor:
    """Closed-form (algebraic, not the unbiased MMSE) inverse of the GAT.

    x = (gain/2 * v)^2 / gain - 3/8*gain - sigma^2/gain
    """
    sigma = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=v.dtype, device=v.device)
    while sigma.dim() < v.dim():
        sigma = sigma.unsqueeze(-1)
    x = (gain * v / 2.0) ** 2 / gain - 0.375 * gain - sigma ** 2 / gain
    return x


# --------------------------------------------------------------------------
# data step - closed-form FFT Wiener solve, no learned parameters
# --------------------------------------------------------------------------

def _kernel_to_otf(kernel: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Zero-pad a small centred spatial kernel to (H,W) and FFT it, with the
    circular shift that keeps the kernel's centre at the origin (so the FFT
    is a true circular-convolution frequency response, not an FFT of an
    off-centre kernel)."""
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros(*kernel.shape[:-2], h, w, dtype=kernel.dtype, device=kernel.device)
    pad[..., :kh, :kw] = kernel
    pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    return torch.fft.fft2(pad)


def identity_otf(shape, h: int, w: int, dtype, device) -> torch.Tensor:
    k = torch.zeros(*shape, 1, 1, dtype=dtype, device=device)
    k[..., 0, 0] = 1.0
    return _kernel_to_otf(k, h, w)


def wiener_data_step(v: torch.Tensor, z: torch.Tensor, otf: torch.Tensor,
                     sigma: torch.Tensor, rho: float) -> torch.Tensor:
    """Closed-form solution of

        argmin_x  (1/(2*sigma^2)) ||v - k*x||^2  +  (rho/2) ||x - z||^2

    x* = F^-1[ (conj(K) F(v) + rho*sigma^2 F(z)) / (|K|^2 + rho*sigma^2) ]

    No learned weights: otf comes from the DBDE-estimated kernel, sigma from
    DBDE, rho from a fixed schedule. Every pixel's update uses the same
    closed-form formula regardless of what is in the image.
    """
    sigma2 = (sigma ** 2).clamp_min(EPS)
    while sigma2.dim() < v.dim():
        sigma2 = sigma2.unsqueeze(-1)
    Fv = torch.fft.fft2(v)
    Fz = torch.fft.fft2(z)
    num = torch.conj(otf) * Fv + rho * sigma2 * Fz
    den = (otf.abs() ** 2 + rho * sigma2).clamp_min(EPS)
    x = torch.fft.ifft2(num / den).real
    return x


# --------------------------------------------------------------------------
# learned proximal denoiser - the only part with content-dependent capacity
# --------------------------------------------------------------------------

class ProxCNN(nn.Module):
    """Tiny residual denoiser. Kept under 200K params on purpose: the cap is
    the mechanism that limits how much content-selective erasure the prox
    step can learn to do, not a tuning knob to relax for better PSNR."""

    def __init__(self, channels: int = 3, width: int = 24, depth: int = 4):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual: identity at init-ish behaviour


# --------------------------------------------------------------------------
# PCIM
# --------------------------------------------------------------------------

class PCIM(nn.Module):
    def __init__(self, n_iters: int = 5, rho0: float = 1.0, rho_scale: float = 2.0,
                 channels: int = 3, prox_width: int = 24, prox_depth: int = 4,
                 illum_clamp: float = 4.0, vst_gain: float = 1.0):
        super().__init__()
        if not (4 <= n_iters <= 6):
            raise ValueError(f"n_iters should be 4-6 per spec, got {n_iters}")
        self.n_iters = n_iters
        self.rho0 = rho0
        self.rho_scale = rho_scale
        self.illum_clamp = illum_clamp
        self.vst_gain = vst_gain
        self.prox = ProxCNN(channels, prox_width, prox_depth)
        self._alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 at init

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    def prox_param_count(self) -> int:
        return sum(p.numel() for p in self.prox.parameters())

    def _hqs(self, v: torch.Tensor, otf: torch.Tensor, sigma: torch.Tensor,
            gate: float) -> torch.Tensor:
        """Run the unrolled recursion. gate=0 -> pure data step (linear,
        content-agnostic). gate=1 -> data step then full prox each iteration."""
        x = v
        for t in range(self.n_iters):
            rho_t = self.rho0 * (self.rho_scale ** t)
            x = wiener_data_step(v, x, otf, sigma, rho_t)
            if gate != 0.0:
                z = self.prox(x)
                x = gate * z + (1.0 - gate) * x
        return x

    def forward(self, y: torch.Tensor, illum_field: torch.Tensor | None = None,
               kernel: torch.Tensor | None = None,
               sigma: torch.Tensor | float = 0.0):
        """
        y: (B,C,H,W) degraded image, float in [0,1]
        illum_field: (B,1,H,W) or (B,H,W) or None (no illumination correction)
        kernel: (B,1,kh,kw) or None (no blur - identity kernel)
        sigma: (B,) or (B,1,1,1) or scalar - DBDE-estimated noise sigma

        Returns (x_full, x_cons), both (B,C,H,W).
        """
        b, c, h, w = y.shape
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
        sigma = sigma.to(y.dtype).reshape(b, *([1] * (y.dim() - 1))) if sigma.numel() == b else sigma

        x = illumination_divide(y, illum_field, self.illum_clamp) if illum_field is not None else y
        v = generalized_anscombe(x, sigma, self.vst_gain)

        if kernel is not None:
            otf = _kernel_to_otf(kernel, h, w)
        else:
            otf = identity_otf((b, c), h, w, y.dtype, y.device)
        if otf.shape[1] == 1 and c > 1:
            otf = otf.expand(b, c, h, w)

        v_full = self._hqs(v, otf, sigma, gate=1.0)
        v_cons = self._hqs(v, otf, sigma, gate=0.0)

        x_full = inverse_generalized_anscombe(v_full, sigma, self.vst_gain)
        x_cons = inverse_generalized_anscombe(v_cons, sigma, self.vst_gain)

        if illum_field is not None:
            x_full = illumination_restore(x_full, illum_field, self.illum_clamp)
            x_cons = illumination_restore(x_cons, illum_field, self.illum_clamp)

        return x_full, x_cons
