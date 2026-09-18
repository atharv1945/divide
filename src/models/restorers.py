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

DEEP_SPECS = {
    "nafnet": dict(
        url="https://github.com/megvii-research/NAFNet",
        weights="NAFNet-SIDD-width64.pth",
        note="Download from the NAFNet repo README (Google Drive) into checkpoints/.",
    ),
    "restormer": dict(
        url="https://github.com/swz30/Restormer",
        weights="real_denoising.pth",
        note="Restormer release assets on GitHub; place in checkpoints/.",
    ),
    "promptir": dict(
        url="https://github.com/va1shn9v/PromptIR",
        weights="promptir_all.ckpt",
        note="Checkpoint linked from the PromptIR repo README.",
    ),
}


def _deep_unavailable(name: str):
    spec = DEEP_SPECS[name]

    def fn(img: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            f"\n{'=' * 70}\nRestorer {name!r} has no weights available.\n"
            f"Repo:    {spec['url']}\nWeights: {spec['weights']}\n{spec['note']}\n"
            f"Place the file under checkpoints/ and re-run.\n"
            f"To run without it, drop {name!r} from the config's `restorers` list.\n"
            f"{'=' * 70}"
        )
    return fn


def load_deep(name: str) -> Restorer | None:
    """Try to build a deep restorer. Returns None if torch/weights are absent."""
    from src.utils.paths import checkpoints_dir
    spec = DEEP_SPECS.get(name)
    if spec is None:
        return None
    path = checkpoints_dir() / spec["weights"]
    if not path.exists():
        return Restorer(name, _deep_unavailable(name), tier="deep", needs_gpu=True)
    try:
        import torch  # noqa: F401
    except ImportError:
        return Restorer(name, _deep_unavailable(name), tier="deep", needs_gpu=True)
    # Real loaders are wired in on the GPU machine; the interface is fixed here
    # so the study script never needs to change.
    return Restorer(name, _deep_unavailable(name), tier="deep", needs_gpu=True)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

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
}


def get_restorer(name: str) -> Restorer:
    if name in CLASSICAL:
        return Restorer(name, CLASSICAL[name], tier="classical")
    deep = load_deep(name)
    if deep is not None:
        return deep
    raise KeyError(
        f"unknown restorer {name!r}. "
        f"classical: {sorted(CLASSICAL)} | deep: {sorted(DEEP_SPECS)}"
    )


def available_restorers(include_deep: bool = False) -> list[str]:
    names = sorted(CLASSICAL)
    if include_deep:
        names += sorted(DEEP_SPECS)
    return names
