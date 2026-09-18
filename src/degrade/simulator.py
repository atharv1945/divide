"""Degradation simulator.

Five families at five severities following the ImageNet-C convention, plus a
`mixed` family that composes two or three at matched severity.

All functions take and return float32 images in [0, 1], shape (H, W, 3).
Every degradation exposes its ground-truth parameters so the DBDE estimator
can be scored against them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

FAMILIES = ["defocus", "motion", "illumination", "noise", "jpeg", "mixed"]
SEVERITIES = [1, 2, 3, 4, 5]

# --------------------------------------------------------------------------
# severity tables  (index 0 == severity 1)
# --------------------------------------------------------------------------
DEFOCUS_RADIUS = [1.0, 2.0, 3.0, 4.5, 6.0]          # px
MOTION_LENGTH = [3, 6, 10, 14, 20]                   # px
ILLUM_EV = [0.30, 0.60, 0.90, 1.20, 1.50]            # stops, signed at random
ILLUM_VIGNETTE = [0.05, 0.10, 0.18, 0.26, 0.35]      # corner falloff fraction
ILLUM_CCT = [250, 500, 900, 1300, 1800]              # K shift magnitude
NOISE_READ = [2.0, 4.0, 7.0, 11.0, 15.0]             # /255
NOISE_SHOT = [0.004, 0.008, 0.014, 0.022, 0.030]     # photon scale
JPEG_QF = [90, 75, 60, 45, 30]


@dataclass
class DegradationParams:
    """Ground-truth degradation parameters for one image."""
    family: str
    severity: int
    blur_kind: str = "none"          # none | defocus | motion
    blur_radius: float = 0.0         # defocus disc radius, px
    blur_length: float = 0.0         # motion length, px
    blur_angle: float = 0.0          # motion angle, radians
    illum_ev: float = 0.0
    illum_vignette: float = 0.0
    illum_cct: float = 0.0
    noise_read: float = 0.0          # sigma in [0,1] units
    noise_shot: float = 0.0
    jpeg_qf: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def vector(self) -> np.ndarray:
        """Normalised parameter vector for regression losses / error metrics."""
        return np.array([
            self.blur_radius / 6.0,
            self.blur_length / 20.0,
            np.cos(2 * self.blur_angle),
            np.sin(2 * self.blur_angle),
            self.illum_ev / 1.5,
            self.illum_vignette / 0.35,
            self.illum_cct / 1800.0,
            self.noise_read / (15.0 / 255.0),
            self.noise_shot / 0.030,
            (100 - self.jpeg_qf) / 70.0,
        ], dtype=np.float32)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def defocus_kernel(radius: float) -> np.ndarray:
    """Disc kernel with anti-aliased rim, normalised to sum 1."""
    radius = max(float(radius), 1e-3)
    r = int(np.ceil(radius)) + 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    k = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else np.ones((1, 1), np.float32)


def motion_kernel(length: float, angle: float) -> np.ndarray:
    """Line kernel of given length (px) and angle (radians), sum 1."""
    length = max(float(length), 1.0)
    n = int(np.ceil(length)) | 1
    n = max(n, 3)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    steps = max(int(length * 4), 8)
    for t in np.linspace(-length / 2.0, length / 2.0, steps):
        x = c + t * np.cos(angle)
        y = c + t * np.sin(angle)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dy in (0, 1):
            for dz in (0, 1):
                yi, xi = y0 + dy, x0 + dz
                if 0 <= yi < n and 0 <= xi < n:
                    k[yi, xi] += (fy if dy else 1 - fy) * (fx if dz else 1 - fx)
    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else np.ones((1, 1), np.float32)


def apply_kernel(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT101)


# --------------------------------------------------------------------------
# illumination
# --------------------------------------------------------------------------

def illumination_field(shape: tuple[int, int], ev: float, vignette: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative field: global gain x low-order tilt x vignette."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(w - 1, 1)) * 2 - 1
    yn = (yy / max(h - 1, 1)) * 2 - 1

    field = np.full((h, w), 2.0 ** ev, np.float32)

    # gentle low-order spatial tilt (degree <= 2), amplitude tied to ev
    a = rng.uniform(-0.12, 0.12) * abs(ev)
    b = rng.uniform(-0.12, 0.12) * abs(ev)
    c = rng.uniform(-0.06, 0.06) * abs(ev)
    field *= (1.0 + a * xn + b * yn + c * xn * yn)

    if vignette > 0:
        r2 = (xn ** 2 + yn ** 2) / 2.0
        field *= (1.0 - vignette * r2)

    return np.clip(field, 0.05, 20.0).astype(np.float32)


def cct_gains(shift_k: float) -> np.ndarray:
    """Crude but monotone colour-temperature gain triplet (R, G, B)."""
    t = float(shift_k) / 1800.0
    return np.array([1.0 + 0.16 * t, 1.0, 1.0 - 0.16 * t], np.float32)


# --------------------------------------------------------------------------
# noise / compression
# --------------------------------------------------------------------------

def add_poisson_gaussian(img: np.ndarray, read_sigma: float, shot_scale: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Signal-dependent sensor noise, drawn from a CONTENT-INDEPENDENT source.

    Shot noise is applied as its Gaussian approximation, sqrt(shot_scale * I),
    rather than by drawing Poisson counts directly. This matters: a direct
    Poisson draw consumes a different number of random values depending on
    pixel intensity, so two images differing only in a small defect region
    would receive *different* noise everywhere. The counterfactual pair that
    L_pres depends on would then be invalid. Drawing standard normals first
    and scaling afterwards keeps the noise field identical across the pair
    while preserving signal dependence.
    """
    out = img.astype(np.float32)
    if shot_scale > 0:
        z = rng.standard_normal(out.shape).astype(np.float32)
        out = out + z * np.sqrt(np.clip(out, 0, 1) * shot_scale)
    if read_sigma > 0:
        out = out + rng.standard_normal(out.shape).astype(np.float32) * read_sigma
    return out


def jpeg_compress(img: np.ndarray, qf: int) -> np.ndarray:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# sampling + application
# --------------------------------------------------------------------------

def sample_params(family: str, severity: int, rng: np.random.Generator) -> DegradationParams:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    i = severity - 1
    p = DegradationParams(family=family, severity=severity)

    if family == "mixed":
        n = int(rng.integers(2, 4))
        picks = list(rng.choice(["defocus", "motion", "illumination", "noise", "jpeg"],
                                size=n, replace=False))
        if "defocus" in picks and "motion" in picks:
            picks.remove(rng.choice(["defocus", "motion"]))
        for sub in picks:
            _fill(p, sub, i, rng)
        return p

    _fill(p, family, i, rng)
    return p


def _fill(p: DegradationParams, family: str, i: int, rng: np.random.Generator) -> None:
    if family == "defocus":
        p.blur_kind = "defocus"
        p.blur_radius = float(DEFOCUS_RADIUS[i])
    elif family == "motion":
        p.blur_kind = "motion"
        p.blur_length = float(MOTION_LENGTH[i])
        p.blur_angle = float(rng.uniform(0, np.pi))
    elif family == "illumination":
        sign = -1.0 if rng.random() < 0.6 else 1.0   # under-exposure more common
        p.illum_ev = sign * float(ILLUM_EV[i])
        p.illum_vignette = float(ILLUM_VIGNETTE[i])
        p.illum_cct = float(rng.choice([-1, 1]) * ILLUM_CCT[i])
    elif family == "noise":
        p.noise_read = float(NOISE_READ[i]) / 255.0
        p.noise_shot = float(NOISE_SHOT[i])
    elif family == "jpeg":
        p.jpeg_qf = int(JPEG_QF[i])


def apply_degradation(img: np.ndarray, p: DegradationParams,
                      rng: np.random.Generator | None = None,
                      clip: bool = True) -> np.ndarray:
    """Apply the physical degradation chain:  Q( L (.) (k * x) ) + n."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.asarray(img, np.float32)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)

    # 1. blur
    if p.blur_kind == "defocus" and p.blur_radius > 0:
        out = apply_kernel(out, defocus_kernel(p.blur_radius))
    elif p.blur_kind == "motion" and p.blur_length > 0:
        out = apply_kernel(out, motion_kernel(p.blur_length, p.blur_angle))

    # 2. illumination (multiplicative, spatially smooth) + colour temperature
    if p.illum_ev != 0.0 or p.illum_vignette != 0.0:
        L = illumination_field(out.shape[:2], p.illum_ev, p.illum_vignette, rng)
        out = out * L[:, :, None]
    if p.illum_cct != 0.0:
        out = out * cct_gains(p.illum_cct)[None, None, :]

    # 3. sensor noise
    if p.noise_read > 0 or p.noise_shot > 0:
        out = add_poisson_gaussian(out, p.noise_read, p.noise_shot, rng)

    if clip:
        out = np.clip(out, 0.0, 1.0)

    # 4. compression last (it quantises whatever reaches the encoder)
    if p.jpeg_qf < 100:
        out = jpeg_compress(np.clip(out, 0, 1), p.jpeg_qf)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def degrade(img: np.ndarray, family: str, severity: int,
            seed: int = 0) -> tuple[np.ndarray, DegradationParams]:
    """Convenience wrapper: sample params for (family, severity) and apply."""
    rng = np.random.default_rng(seed)
    p = sample_params(family, severity, rng)
    return apply_degradation(img, p, rng), p


def degrade_pair(img_a: np.ndarray, img_b: np.ndarray, family: str, severity: int,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray, DegradationParams]:
    """Degrade two images with IDENTICAL params and noise seed.

    This is what makes the counterfactual defect-preservation loss computable:
    pass (image_with_anomaly, image_without_anomaly).
    """
    p = sample_params(family, severity, np.random.default_rng(seed))
    ya = apply_degradation(img_a, p, np.random.default_rng(seed + 1))
    yb = apply_degradation(img_b, p, np.random.default_rng(seed + 1))
    return ya, yb, p
