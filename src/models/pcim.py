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

SARG (src/models/sarg.py) blends between the two per-pixel using a learned
protection mask: x = (1-m)*x_full + m*x_cons. That mask is where the
full/conservative tradeoff is decided, and it varies by region.

An earlier version of this module also exposed a single learned SCALAR gate
(`alpha`, sigmoid-squashed) meant to blend x_full/x_cons into one deployable
output. It was removed: a global per-image gate is redundant with - and
would compete with - SARG's spatial gating, since how aggressively to
restore should vary by region, not be one number per image. `forward()`
always returns the x_full / x_cons pair; blending is SARG's job, not
PCIM's.
"""
from __future__ import annotations

import math

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
    step can learn to do, not a tuning knob to relax for better PSNR.

    The final conv layer's weight and bias are explicitly zeroed, so
    net(x) is exactly 0 and forward(x) is exactly x at init - a true
    identity, not "identity-ish". This was previously just a comment, not
    code: PyTorch's default conv init does not produce a small output, and
    x_full applies this block 5 times per forward pass (gate=1 in
    PCIM._hqs) - an untrained, non-zero residual compounded 5x was
    measurably destroying the image before any training happened (found by
    actually evaluating step-0 x_full on real data, not by inspecting the
    module in isolation: DRemR -8.11, PSNR 13.16dB, indistinguishable from
    a fully-trained-but-broken-physics run's final numbers). Zero-init is
    standard practice for residual branches precisely because of this
    failure mode."""

    def __init__(self, channels: int = 3, width: int = 24, depth: int = 4):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual: EXACTLY identity at init


# --------------------------------------------------------------------------
# PCIM
# --------------------------------------------------------------------------

class PCIM(nn.Module):
    def __init__(self, n_iters: int = 5, rho0: float = 1.0, rho_scale: float = 2.0,
                 channels: int = 3, prox_width: int = 24, prox_depth: int = 4,
                 illum_clamp: float = 4.0, vst_gain: float = 1.0,
                 nsr_floor: float = 0.01):
        super().__init__()
        if not (4 <= n_iters <= 6):
            raise ValueError(f"n_iters should be 4-6 per spec, got {n_iters}")
        self.n_iters = n_iters
        self.rho0 = rho0
        self.rho_scale = rho_scale
        self.illum_clamp = illum_clamp
        self.vst_gain = vst_gain
        # Floor on the sigma fed to the WIENER regularisation term specifically
        # (rho*sigma^2 in wiener_data_step's denominator) - NOT on the sigma
        # used by the VST, which needs the true estimated noise level to
        # variance-stabilise correctly. DBDE's noise estimate is frequently
        # near its own numerical floor on real images (median ~2e-4 in one
        # overnight run), which makes rho*sigma^2 negligible next to |otf|^2
        # and lets Wiener behave like an unregularised inverse filter -
        # catastrophic at any frequency where the kernel has near-zero
        # response. nsr_floor is a floor on sigma^2 (matching "nsr" in the
        # classical sense: the term added to |K|^2 in the denominator), so
        # the effective sigma used for Wiener is max(sigma, sqrt(nsr_floor)).
        # Config-driven (model.nsr_floor in configs/train_pcim_*.yaml) -
        # deliberately not derived from the estimate alone, since the
        # estimate is exactly what's being distrusted here.
        self.nsr_floor = nsr_floor
        self.prox = ProxCNN(channels, prox_width, prox_depth)

    def prox_param_count(self) -> int:
        return sum(p.numel() for p in self.prox.parameters())

    def _hqs(self, v: torch.Tensor, otf: torch.Tensor | None, sigma: torch.Tensor,
            gate: float) -> torch.Tensor:
        """Run the unrolled recursion. gate=0 -> pure data step (linear,
        content-agnostic). gate=1 -> data step then full prox each iteration.

        otf=None means "no blur was detected - skip the Wiener data step
        entirely" (see forward()'s docstring on why this must be an actual
        skip, not a deconvolution against an identity kernel)."""
        x = v
        for t in range(self.n_iters):
            if otf is not None:
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
        kernel: (B,1,kh,kw), or None if no blur was detected. None means
            SKIP the Wiener data step entirely - "doing nothing" (beyond
            illumination/VST) must always be an available output, not
            approximated by deconvolving against an identity kernel. The two
            are not equivalent under noise: an identity-kernel Wiener pass
            still divides by (1 + rho*sigma^2) every iteration, which is
            harmless at a sane sigma but is one more opportunity for a bad
            sigma estimate to do damage for no physical reason, since there
            is nothing to deconvolve. Callers (train_pcim.py,
            divide_restorer.py) pass kernel=None precisely when DBDE's
            blur_kind is "none" - see src/dbde/estimator.py's blur-decision
            functions.
        sigma: (B,) or (B,1,1,1) or scalar - DBDE-estimated noise sigma

        Returns (x_full, x_cons), both (B,C,H,W).
        """
        b, c, h, w = y.shape
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
        sigma = sigma.to(y.dtype).reshape(b, *([1] * (y.dim() - 1))) if sigma.numel() == b else sigma
        sigma_wiener = sigma.clamp(min=self.nsr_floor ** 0.5)

        x = illumination_divide(y, illum_field, self.illum_clamp) if illum_field is not None else y
        v = generalized_anscombe(x, sigma, self.vst_gain)

        if kernel is not None:
            otf = _kernel_to_otf(kernel, h, w)
            if otf.shape[1] == 1 and c > 1:
                otf = otf.expand(b, c, h, w)
        else:
            otf = None

        v_full = self._hqs(v, otf, sigma_wiener, gate=1.0)
        v_cons = self._hqs(v, otf, sigma_wiener, gate=0.0)

        x_full = inverse_generalized_anscombe(v_full, sigma, self.vst_gain)
        x_cons = inverse_generalized_anscombe(v_cons, sigma, self.vst_gain)

        if illum_field is not None:
            x_full = illumination_restore(x_full, illum_field, self.illum_clamp)
            x_cons = illumination_restore(x_cons, illum_field, self.illum_clamp)

        return x_full, x_cons


_PHYSICS_FAIL_EXPLANATION = (
    "x_cons has no learned component - illumination divide, VST, "
    "closed-form Wiener, inverse VST only. A DRemR this negative "
    "means the PURE PHYSICS PATH is destroying the image, not the "
    "learned prox or L_pres. Do not retune loss weights for this - "
    "see src/experiments/bisect_pcim.py to isolate which physics "
    "stage is responsible (illumination / VST / Wiener) before "
    "changing anything.\n"
    f"{'=' * 70}"
)


def assert_physics_sane(x_cons_dremr: float, floor: float = -0.5) -> None:
    """x_cons is the alpha=0 path: illumination divide, VST, closed-form
    Wiener, inverse VST - nothing learned. It has no mechanism to make
    training-driven excuses for being wrong, so if its DRemR on held-out
    data is far below "did nothing" (DRemR=0 by construction), that is
    always a physics-chain bug, not a training issue, and it should stop
    the run loudly rather than be silently absorbed by the learned prox
    over the following hours - which is exactly what happened before this
    check existed. Called from train_pcim.py's evaluate() every eval_every
    steps.

    This is TIER 1 of assert_physics_sane_stratified below - kept as a
    standalone single-float function too since most callers (a single
    scalar eval-set mean) don't have or need the per-example breakdown the
    stratified version requires.
    """
    if x_cons_dremr < floor:
        raise RuntimeError(
            f"\n{'=' * 70}\n"
            f"PHYSICS SANITY CHECK FAILED (aggregate mean): x_cons DRemR = "
            f"{x_cons_dremr:.3f}, below the floor of {floor}.\n"
            f"{_PHYSICS_FAIL_EXPLANATION}"
        )


def assert_physics_sane_stratified(
    records: list[tuple[str, int, float]],
    aggregate_floor: float = -0.5,
    cell_floor: float = -0.5,
    cell_violation_fraction: float = 0.34,
    cell_min_n: int = 3,
    singleton_floor: float = -2.0,
) -> None:
    """Three-tier physics-sanity guard over x_cons DRemR - replaces a
    single aggregate-mean check that provably has a blind spot: an eval set
    mixing easy and hard (family, severity) combinations can sit
    comfortably above `aggregate_floor` on average while individual
    examples, or one whole cell, sit far below it. Measured directly (see
    src/experiments/eval_pcim_holdout.py's physics_guard_check): 24-example
    training eval mean -0.021, 150-example held-out mean -0.065, both
    nowhere near -0.5, while 6/150 individual examples were below -0.5
    (worst -4.19), 3 of them from a single (carpet, defocus, severity-1)
    cell that should have been among the EASIEST in the set.

    Tier 1 - AGGREGATE MEAN (`aggregate_floor`, same check as
    assert_physics_sane): catches total, uniform catastrophe.

    Tier 2 - PER-(family, severity) CELL, `cell_floor` AND
    `cell_violation_fraction` (both required, not either): catches a
    systematic failure concentrated in one combination that the aggregate
    dilutes away, while staying stable against single-example ratio noise
    (metrics.core.degradation_removal_ratio's docstring documents why a
    lone example's DRemR can swing hard on its own) because it still
    averages several examples per cell and additionally requires a real
    fraction of them to individually violate, not just one bad outlier
    dragging a small cell's mean down. Cells smaller than `cell_min_n` are
    skipped here (left to tier 3 instead) - found while testing: without a
    minimum, a cell of size 1 makes "mean" and "fraction violating"
    degenerate into the exact same tight per-example floor this tier was
    designed to avoid (one bad example is both 100% of the cell AND its
    whole mean).

    Tier 3 - PER-EXAMPLE, `singleton_floor` (deliberately far looser than
    -0.5, NOT -0.5 itself): a tripwire for one genuine catastrophic
    example that a cell average could still dilute if the rest of the cell
    is well-behaved. -2.0 is well past anything the ratio-instability
    mechanism alone produced even at the smallest denominators measured
    (the mildest third's worst single case was -0.60) - a per-example
    floor this loose should only fire on a real failure, not metric noise,
    which is exactly why a TIGHT per-example floor (e.g. at -0.5, the same
    value as the aggregate/cell floors) was rejected: it would false-
    trigger on exactly the noise this project spent effort characterising.

    `records` is a flat list of (family, severity, x_cons_dremr) - x_cons
    only, same reasoning as assert_physics_sane: it has no learned
    component, so a violation is always a physics-chain bug, never a
    training-driven excuse.
    """
    finite = [(f, s, d) for f, s, d in records if math.isfinite(d)]
    if not finite:
        return

    mean_all = sum(d for _, _, d in finite) / len(finite)
    if mean_all < aggregate_floor:
        raise RuntimeError(
            f"\n{'=' * 70}\n"
            f"PHYSICS SANITY CHECK FAILED (tier 1, aggregate mean over "
            f"{len(finite)} examples): x_cons DRemR mean = {mean_all:.3f}, "
            f"below the floor of {aggregate_floor}.\n"
            f"{_PHYSICS_FAIL_EXPLANATION}"
        )

    cells: dict[tuple[str, int], list[float]] = {}
    for f, s, d in finite:
        cells.setdefault((f, s), []).append(d)
    for (family, severity), vals in sorted(cells.items()):
        if len(vals) < cell_min_n:
            continue
        cell_mean = sum(vals) / len(vals)
        frac_violating = sum(1 for v in vals if v < cell_floor) / len(vals)
        if cell_mean < cell_floor and frac_violating >= cell_violation_fraction:
            raise RuntimeError(
                f"\n{'=' * 70}\n"
                f"PHYSICS SANITY CHECK FAILED (tier 2, cell): "
                f"family={family!r} severity={severity}, mean x_cons DRemR = "
                f"{cell_mean:.3f} (< {cell_floor}), {frac_violating:.0%} of "
                f"{len(vals)} examples in this cell individually violating "
                f"(>= {cell_violation_fraction:.0%} required).\n"
                f"{_PHYSICS_FAIL_EXPLANATION}"
            )

    worst_family, worst_severity, worst_d = min(finite, key=lambda r: r[2])
    if worst_d < singleton_floor:
        raise RuntimeError(
            f"\n{'=' * 70}\n"
            f"PHYSICS SANITY CHECK FAILED (tier 3, singleton): "
            f"family={worst_family!r} severity={worst_severity}, "
            f"x_cons DRemR = {worst_d:.3f}, below the singleton floor of "
            f"{singleton_floor}.\n"
            f"{_PHYSICS_FAIL_EXPLANATION}"
        )
