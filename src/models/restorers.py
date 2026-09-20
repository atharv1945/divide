"""Restoration baselines.

Two tiers:

  classical  - CLAHE, MSRCR, homomorphic, Wiener, Richardson-Lucy, BM3D-ish,
               bilateral, unsharp. Pure CPU, always available, zero downloads.
  deep       - NAFNet / Restormer / diffusion. Loaded lazily; if weights are
               absent the registry reports the method as unavailable with the
               exact URL rather than silently skipping it.

Every restorer has the same signature:  restore(img_float01) -> img_float01
so the DRR study can iterate over them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

EPS = 1e-8


@dataclass
class Restorer:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    tier: str = "classical"
    needs_gpu: bool = False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = self.fn(np.asarray(img, np.float32))
        return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# trivial
# --------------------------------------------------------------------------

def identity(img: np.ndarray) -> np.ndarray:
    return img.copy()


# --------------------------------------------------------------------------
# classical restorers
# --------------------------------------------------------------------------

def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def msrcr(img: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """Multi-scale Retinex with colour restoration."""
    x = np.clip(img, 1e-3, 1.0)
    logx = np.log(x)
    acc = np.zeros_like(x)
    for s in sigmas:
        blur = cv2.GaussianBlur(x, (0, 0), sigmaX=s, sigmaY=s,
                                borderType=cv2.BORDER_REFLECT101)
        acc += logx - np.log(np.clip(blur, 1e-3, None))
    r = acc / len(sigmas)
    ssum = x.sum(axis=2, keepdims=True) + EPS
    crf = 1.0 * (np.log(1.0 + 46.0 * x / ssum))
    out = crf * r
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return (out - lo) / (hi - lo + EPS)


def homomorphic(img: np.ndarray, sigma_frac: float = 1 / 12,
                gain_low: float = 0.5, gain_high: float = 1.5) -> np.ndarray:
    g = np.clip(img, 1e-3, 1.0)
    logg = np.log(g)
    h, w = g.shape[:2]
    sigma = max(min(h, w) * sigma_frac, 3.0)
    low = cv2.GaussianBlur(logg, (0, 0), sigmaX=sigma, sigmaY=sigma,
                           borderType=cv2.BORDER_REFLECT101)
    high = logg - low
    out = np.exp(gain_low * low + gain_high * high)
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return (out - lo) / (hi - lo + EPS)


def wiener_deconv(img: np.ndarray, kernel: np.ndarray | None = None,
                  nsr: float = 0.01) -> np.ndarray:
    """FFT-domain Wiener deconvolution. Estimates the kernel if none given."""
    from src.dbde.estimator import estimate

    if kernel is None:
        kernel = estimate(img).kernel()

    h, w = img.shape[:2]
    K = np.fft.fft2(kernel, s=(h, w))
    K = K * np.exp(-2j * np.pi * (
        np.fft.fftfreq(h)[:, None] * (kernel.shape[0] // 2) +
        np.fft.fftfreq(w)[None, :] * (kernel.shape[1] // 2)))
    W = np.conj(K) / (np.abs(K) ** 2 + nsr)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.fft2(img[:, :, c]) * W))
    return out


def richardson_lucy(img: np.ndarray, kernel: np.ndarray | None = None,
                    iters: int = 12) -> np.ndarray:
    from src.dbde.estimator import estimate
    if kernel is None:
        kernel = estimate(img).kernel()
    k = kernel.astype(np.float32)
    kf = k[::-1, ::-1].copy()

    out = np.clip(img, 1e-3, 1.0).copy()
    obs = np.clip(img, 1e-3, 1.0)
    for _ in range(iters):
        conv = np.stack([cv2.filter2D(out[:, :, c], -1, k,
                                      borderType=cv2.BORDER_REFLECT101)
                         for c in range(3)], axis=2)
        ratio = obs / np.clip(conv, 1e-3, None)
        corr = np.stack([cv2.filter2D(ratio[:, :, c], -1, kf,
                                      borderType=cv2.BORDER_REFLECT101)
                         for c in range(3)], axis=2)
        out = np.clip(out * corr, 0, 1)
    return out


def nlm_denoise(img: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Non-local means - a strong classical denoiser, BM3D's nearest sibling.

    Uses the estimated noise level so the strength adapts to the input, which
    is the fair setting for a baseline.
    """
    from src.dbde.estimator import estimate_noise_sigma
    if strength is None:
        strength = float(np.clip(estimate_noise_sigma(img) * 255 * 1.2, 2, 25))
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(u8, None, strength, strength, 7, 21)
    return out.astype(np.float32) / 255.0


def bilateral(img: np.ndarray, d: int = 9, sc: float = 0.1,
              ss: float = 7.0) -> np.ndarray:
    return cv2.bilateralFilter(img.astype(np.float32), d, sc, ss)


def unsharp(img: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma,
                            borderType=cv2.BORDER_REFLECT101)
    return img + amount * (img - blur)


def gaussian_denoise(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma,
                            borderType=cv2.BORDER_REFLECT101)


def classical_pipeline(img: np.ndarray) -> np.ndarray:
    """A sensible classical stack: illumination -> denoise -> deblur.

    This is the honest classical competitor. It may well beat the deep
    restorers on defect retention, which would itself be a finding worth
    reporting rather than hiding.
    """
    from src.dbde.estimator import estimate, correct_illumination
    est = estimate(img)
    out = correct_illumination(img, est.illum_field)
    if est.noise_sigma > 3 / 255:
        out = nlm_denoise(out)
    if est.blur_kind != "none":
        out = wiener_deconv(out, est.kernel(), nsr=0.02)
    return out


# --------------------------------------------------------------------------
# deep restorers (lazy, fail loudly)
# --------------------------------------------------------------------------

def _deep_specs() -> dict:
    from src.models.deep_restorers import REPO_SPECS
    return REPO_SPECS


# Backward-compatible alias: repo/weights info for each deep restorer,
# sourced from deep_restorers.REPO_SPECS so there is one place that knows
# the download URLs.
DEEP_SPECS = _deep_specs()


def _deep_unavailable(name: str, err: Exception):
    def fn(img: np.ndarray) -> np.ndarray:
        raise err
    return fn


def load_deep(name: str) -> Restorer | None:
    """Try to build a deep restorer.

    Returns a Restorer whose call raises RuntimeError with the exact
    download URL if weights/repo are absent - never None for a name that IS
    a known deep restorer, and never a silent skip. Returns None only if
    `name` isn't a deep restorer at all (so the caller can report "unknown
    restorer" instead).
    """
    if name not in DEEP_SPECS:
        return None
    from src.models.deep_restorers import build_deep_restorer
    try:
        return build_deep_restorer(name)
    except RuntimeError as e:
        return Restorer(name, _deep_unavailable(name, e), tier="deep", needs_gpu=True)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _wiener_dbde_nsr(img: np.ndarray) -> np.ndarray:
    from src.models.wiener_ablation import wiener_dbde_nsr
    return wiener_dbde_nsr(img)


def _wiener_hqs_rawpixel(img: np.ndarray) -> np.ndarray:
    from src.models.wiener_ablation import wiener_hqs_rawpixel
    return wiener_hqs_rawpixel(img)


def _wiener_hqs_vst(img: np.ndarray) -> np.ndarray:
    from src.models.wiener_ablation import wiener_hqs_vst
    return wiener_hqs_vst(img)


def _wiener_hqs_vst_nofloor(img: np.ndarray) -> np.ndarray:
    from src.models.wiener_ablation import wiener_hqs_vst_nofloor
    return wiener_hqs_vst_nofloor(img)


CLASSICAL: dict[str, Callable] = {
    "identity": identity,
    "clahe": clahe,
    "msrcr": msrcr,
    "homomorphic": homomorphic,
    "gaussian": gaussian_denoise,
    "bilateral": bilateral,
    "nlm": nlm_denoise,
    "wiener": wiener_deconv,
    "richardson_lucy": richardson_lucy,
    "unsharp": unsharp,
    "classical_pipeline": classical_pipeline,
    # Bridging ablation: classical Wiener (config A = "wiener" above) ->
    # PCIM's x_cons, one variable at a time - see
    # src/models/wiener_ablation.py's module docstring for the full
    # rationale (testing whether defect erosion is a property of
    # regularization strength, not method class).
    "wiener_dbde_nsr": _wiener_dbde_nsr,             # config B
    "wiener_hqs_rawpixel": _wiener_hqs_rawpixel,     # config C
    "wiener_hqs_vst": _wiener_hqs_vst,               # config D (= x_cons)
    "wiener_hqs_vst_nofloor": _wiener_hqs_vst_nofloor,  # config E (reverse direction)
}


def get_restorer(name: str) -> Restorer:
    if name in CLASSICAL:
        return Restorer(name, CLASSICAL[name], tier="classical")
    if name == "divide":
        from src.models.divide_restorer import build_divide_restorer
        return build_divide_restorer()
    if name == "restormer_deblur":
        # Not a DEEP_SPECS entry - it composes restormer_motion_deblur and
        # restormer_defocus_deblur, dispatching per image by DBDE's
        # blur_kind estimate (see deep_restorers.build_restormer_deblur_
        # restorer's docstring). Same deferred-error contract as
        # load_deep() below: get_restorer() itself never raises, the
        # RuntimeError fires when the returned Restorer is CALLED.
        from src.models.deep_restorers import build_restormer_deblur_restorer
        try:
            return build_restormer_deblur_restorer()
        except RuntimeError as e:
            return Restorer(name, _deep_unavailable(name, e), tier="deep", needs_gpu=False)
    deep = load_deep(name)
    if deep is not None:
        return deep
    raise KeyError(
        f"unknown restorer {name!r}. "
        f"classical: {sorted(CLASSICAL)} | deep: {sorted(DEEP_SPECS)} | divide | restormer_deblur"
    )


def available_restorers(include_deep: bool = False) -> list[str]:
    names = sorted(CLASSICAL)
    if include_deep:
        names += sorted(DEEP_SPECS) + ["divide", "restormer_deblur"]
    return names
