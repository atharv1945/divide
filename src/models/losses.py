"""Training losses for PCIM.

L_pres is the central novelty and the most fragile: it needs a counterfactual
pair (image_with_anomaly, image_without_anomaly) degraded with IDENTICAL
parameters and an IDENTICAL noise field - that's what src.degrade.simulator
.degrade_pair() guarantees (see its docstring and the bug it fixed). Get that
pairing wrong and L_pres degrades silently into noise rather than a training
curve that visibly fails, because the two restored images stop being
comparable for reasons that have nothing to do with the anomaly.

All four losses are plain functions over already-computed tensors - none of
them call PCIM internally - so they're testable independently of the model
and composable however drr_study/train scripts want.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.pcim import _kernel_to_otf, identity_otf


def reconstruction_loss(x_hat: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """L_rec - plain L1 to the clean target."""
    return F.l1_loss(x_hat, x_clean)


def degradation_consistency_loss(x_hat: torch.Tensor, y_observed: torch.Tensor,
                                 illum_field: torch.Tensor | None = None,
                                 kernel: torch.Tensor | None = None,
                                 illum_clamp: float = 4.0) -> torch.Tensor:
    """L_deg - re-apply the DETERMINISTIC part of the estimated forward
    degradation (blur, illumination) to the restoration and compare against
    what was actually observed. The additive noise term is left out: it's
    zero-mean, so including it would just add variance to the loss without
    changing its minimiser, and DBDE's noise estimate is not itself part of
    the forward operator being checked here (the VST already consumes it).

    x_hat: (B,C,H,W) reconstruction (x_full, typically)
    y_observed: (B,C,H,W) the actual degraded image the model saw
    """
    b, c, h, w = x_hat.shape
    x = x_hat
    if kernel is not None:
        otf = _kernel_to_otf(kernel, h, w)
        if otf.shape[1] == 1 and c > 1:
            otf = otf.expand(b, c, h, w)
        x = torch.fft.ifft2(torch.fft.fft2(x) * otf).real
    if illum_field is not None:
        field = illum_field.unsqueeze(1) if illum_field.dim() == 3 else illum_field
        x = x * field.clamp(1.0 / illum_clamp, illum_clamp)
    return F.l1_loss(x, y_observed)


def frequency_loss(x_hat: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """L_freq - L1 on FFT magnitude. Spatial L1 alone under-weights fine
    texture/edges relative to their visual importance; this penalises
    magnitude-spectrum mismatches directly."""
    fx = torch.fft.fft2(x_hat).abs()
    fc = torch.fft.fft2(x_clean).abs()
    return F.l1_loss(fx, fc)


def preservation_loss(x_tilde_a: torch.Tensor, x_tilde_0: torch.Tensor,
                      x_a: torch.Tensor, x_0: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """L_pres = || (x~_a - x~_0) - (x_a - x_0) ||_1, averaged over mask==1.

    x_tilde_a, x_tilde_0: restorer output for the WITH-anomaly / WITHOUT-
        anomaly member of a counterfactual pair (same degradation params,
        same noise field - see degrade_pair()).
    x_a, x_0: the corresponding CLEAN (pre-degradation) images.
    mask: (B,H,W) or (B,1,H,W), 1 == inside the anomaly.

    Zero means the restorer changed the defect region by exactly as much as
    the clean images differ there - i.e. it neither erased nor hallucinated
    defect signal. This is evaluated only inside the mask: L_pres says
    nothing about fidelity elsewhere (L_rec covers that).
    """
    if mask.dim() == x_tilde_a.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(x_tilde_a.dtype)

    got = x_tilde_a - x_tilde_0
    want = x_a - x_0
    diff = (got - want).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def divide_loss(x_full: torch.Tensor, x_clean: torch.Tensor, y_observed: torch.Tensor,
                x_tilde_a: torch.Tensor, x_tilde_0: torch.Tensor,
                x_a: torch.Tensor, x_0: torch.Tensor, mask: torch.Tensor,
                illum_field: torch.Tensor | None = None,
                kernel: torch.Tensor | None = None,
                w_rec: float = 1.0, w_deg: float = 0.5,
                w_freq: float = 0.1, w_pres: float = 1.0) -> dict[str, torch.Tensor]:
    """Weighted sum of all four losses. Returns the components too, so a
    training loop can log them separately - if L_pres stops moving while
    L_rec keeps improving, that's the erasure failure mode this project
    exists to prevent, and it's invisible if you only log the total."""
    l_rec = reconstruction_loss(x_full, x_clean)
    l_deg = degradation_consistency_loss(x_full, y_observed, illum_field, kernel)
    l_freq = frequency_loss(x_full, x_clean)
    l_pres = preservation_loss(x_tilde_a, x_tilde_0, x_a, x_0, mask)
    total = w_rec * l_rec + w_deg * l_deg + w_freq * l_freq + w_pres * l_pres
    return dict(total=total, rec=l_rec, deg=l_deg, freq=l_freq, pres=l_pres)
