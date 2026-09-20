"""Bridges classical Wiener deconvolution to PCIM's x_cons one variable at
a time, to test a specific mechanistic hypothesis: that classical Wiener's
severe defect erosion (scratch relative DRR 0.093, drr_study_deblur) and
PCIM's own x_cons NOT showing comparable erosion are explained by
regularization strength, not by architecture (learned vs. classical,
single-shot vs. unrolled, pixel vs. VST domain).

Five configurations, each changing exactly one thing from the previous:

  A. classical_wiener_flat_nsr   - restorers.wiener_deconv as-is: one
                                    closed-form division, nsr is a FLAT
                                    constant (default 0.01), raw pixel
                                    domain. (Not redefined here - this IS
                                    the existing "wiener" restorer; reuse
                                    that name in any study using these.)
  B. classical_wiener_dbde_nsr   - same one-shot division, but nsr is
                                    DERIVED per image from DBDE's estimated
                                    sigma, floored the way PCIM floors it
                                    (rho0 * max(sigma, sqrt(nsr_floor))^2)
                                    instead of a blanket constant.
  C. wiener_hqs_rawpixel         - same floored-sigma regularization as B,
                                    but iterated through PCIM's unrolled
                                    HQS recursion (rho_t = rho0 *
                                    rho_scale^t, GROWING each iteration)
                                    instead of one division. Still raw
                                    pixel domain - PCIM._hqs called
                                    directly on the image, no VST, no
                                    illumination correction.
  D. wiener_hqs_vst              - same as C, but wrapped in PCIM's full
                                    illumination-divide -> VST ->
                                    inverse-VST -> illumination-restore
                                    pipeline. This IS x_cons (gate=0, prox
                                    never touched) at PCIM's normal
                                    nsr_floor - not a new mechanism, just
                                    PCIM.forward() with the config's
                                    default nsr_floor.
  E. wiener_hqs_vst_nofloor      - same as D, but nsr_floor set near zero,
                                    so sigma_wiener is no longer floored -
                                    the REVERSE-direction test: if this
                                    erodes like classical Wiener (A), that
                                    confirms regularization strength (not
                                    architecture) is what protects x_cons.

All five are content-agnostic in the same structural sense x_cons always
is (see test_hqs_without_prox_is_linear) - none of them touch a learned
parameter. The hypothesis under test is about regularization STRENGTH,
not about whether a component is learned.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


def _dbde_kernel_and_sigma(img: np.ndarray):
    from src.dbde.estimator import estimate
    est = estimate(img)
    kernel = est.kernel() if est.blur_kind != "none" else None
    sigma = max(est.noise_sigma, 1e-4)
    return kernel, sigma


def wiener_dbde_nsr(img: np.ndarray, rho0: float = 1.0, nsr_floor: float = 0.01) -> np.ndarray:
    """Config B: one-shot classical Wiener, nsr derived from DBDE's own
    (floored) sigma estimate instead of a flat constant."""
    from src.models.restorers import wiener_deconv
    kernel, sigma = _dbde_kernel_and_sigma(img)
    if kernel is None:
        return img.copy()
    sigma_floored = max(sigma, nsr_floor ** 0.5)
    nsr = rho0 * sigma_floored ** 2
    return wiener_deconv(img, kernel=kernel, nsr=nsr)


@torch.no_grad()
def wiener_hqs_rawpixel(img: np.ndarray, n_iters: int = 5, rho0: float = 1.0,
                        rho_scale: float = 2.0, nsr_floor: float = 0.01) -> np.ndarray:
    """Config C: same regularization as config B, but through PCIM's
    unrolled HQS recursion (growing rho) instead of one division - still
    raw pixel domain, no VST/illumination."""
    from src.models.pcim import PCIM, _kernel_to_otf
    kernel, sigma = _dbde_kernel_and_sigma(img)
    if kernel is None:
        return img.copy()

    model = PCIM(n_iters=n_iters, rho0=rho0, rho_scale=rho_scale, nsr_floor=nsr_floor)
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()
    _, _, h, w = x.shape
    k = torch.from_numpy(np.ascontiguousarray(kernel)).float().unsqueeze(0).unsqueeze(0)
    otf = _kernel_to_otf(k, h, w)
    if otf.shape[1] == 1 and x.shape[1] > 1:
        otf = otf.expand(x.shape[0], x.shape[1], h, w)
    sigma_t = torch.tensor([sigma])
    sigma_wiener = sigma_t.clamp(min=model.nsr_floor ** 0.5)

    out = model._hqs(x, otf, sigma_wiener, gate=0.0)
    return out.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def _x_cons(img: np.ndarray, n_iters: int = 5, rho0: float = 1.0,
           rho_scale: float = 2.0, nsr_floor: float = 0.01) -> np.ndarray:
    from src.dbde.estimator import estimate
    from src.models.pcim import PCIM

    est = estimate(img)
    model = PCIM(n_iters=n_iters, rho0=rho0, rho_scale=rho_scale, nsr_floor=nsr_floor)
    model.eval()

    y = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()
    illum = torch.from_numpy(np.ascontiguousarray(est.illum_field)).unsqueeze(0).unsqueeze(0).float()
    kernel = (torch.from_numpy(np.ascontiguousarray(est.kernel())).unsqueeze(0).unsqueeze(0).float()
             if est.blur_kind != "none" else None)
    sigma = torch.tensor([max(est.noise_sigma, 1e-4)])

    _, x_cons = model(y, illum_field=illum, kernel=kernel, sigma=sigma)
    return x_cons.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()


def wiener_hqs_vst(img: np.ndarray) -> np.ndarray:
    """Config D: config C wrapped in PCIM's full illumination/VST pipeline
    - this IS x_cons at the model's normal (configs/train_pcim_*.yaml)
    nsr_floor of 0.01."""
    return _x_cons(img, nsr_floor=0.01)


def wiener_hqs_vst_nofloor(img: np.ndarray) -> np.ndarray:
    """Config E: x_cons with nsr_floor disabled (set near zero) - the
    reverse-direction test. sigma_wiener is then whatever DBDE's raw
    estimate says, unfloored - frequently near its own numerical floor on
    real images (documented median ~2e-4 in pcim.py's nsr_floor comment),
    which makes rho*sigma^2 negligible next to |otf|^2 near the kernel's
    spectral nulls - an effectively unregularized inverse filter."""
    return _x_cons(img, nsr_floor=1e-8)
