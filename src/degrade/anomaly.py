"""Synthetic anomaly generation.

Produces (image_with_anomaly, binary_mask) from a clean normal image, using
three anomaly families that between them cover the realistic failure modes:

  texture  - DTD patch blended through a Perlin-noise mask (DRAEM lineage)
  scratch  - thin Bezier stroke, 1-3 px wide  (the hardest case for a denoiser)
  blob     - smooth low-contrast region       (subtle discolouration / stain)

The mask is exact, which is what lets us compute the Defect Retention Ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ANOMALY_KINDS = ["texture", "scratch", "blob"]


@dataclass
class AnomalySpec:
    kind: str
    area_frac: float          # fraction of image pixels covered
    contrast: float           # nominal amplitude of the perturbation


# --------------------------------------------------------------------------
# perlin noise (self-contained, no external dep)
# --------------------------------------------------------------------------

def _perlin(shape: tuple[int, int], res: tuple[int, int],
            rng: np.random.Generator) -> np.ndarray:
    """Classic 2-D Perlin noise in roughly [-1, 1]."""
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    h, w = shape
    ry, rx = res
    # work on a grid that divides evenly, then resize
    gh, gw = ry * int(np.ceil(h / ry)), rx * int(np.ceil(w / rx))

    d = (gh // ry, gw // rx)
    grid = np.mgrid[0:ry:ry / gh, 0:rx:rx / gw].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * rng.random((ry + 1, rx + 1))
    grads = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = grads[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = grads[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = grads[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = grads[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    out = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    return out[:h, :w].astype(np.float32)


def perlin_mask(shape: tuple[int, int], rng: np.random.Generator,
                target_frac: float = 0.02) -> np.ndarray:
    """Binary-ish Perlin mask thresholded to approximately target_frac area."""
    h, w = shape
    res = (int(2 ** rng.integers(1, 4)), int(2 ** rng.integers(1, 4)))
    n = _perlin((h, w), res, rng)
    n = (n - n.min()) / (float(n.max() - n.min()) + 1e-8)
    thr = np.quantile(n, 1.0 - float(np.clip(target_frac, 1e-4, 0.5)))
    m = (n >= thr).astype(np.float32)
    # keep a single connected blob so area is well-controlled
    num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if num > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == biggest).astype(np.float32)
    return m


# --------------------------------------------------------------------------
# texture source
# --------------------------------------------------------------------------

class TextureBank:
    """Lazy loader over DTD images. Falls back to procedural noise if absent."""

    def __init__(self, dtd_dir: Path | None = None, limit: int = 400):
        self.files: list[Path] = []
        if dtd_dir is not None:
            img_dir = Path(dtd_dir) / "images"
            if img_dir.exists():
                for p in sorted(img_dir.rglob("*.jpg"))[:limit]:
                    self.files.append(p)

    @property
    def available(self) -> bool:
        return len(self.files) > 0

    def sample(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        h, w = shape
        if self.available:
            p = self.files[int(rng.integers(len(self.files)))]
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # procedural fallback: coloured band-limited noise
        n = rng.random((max(h // 8, 2), max(w // 8, 2), 3)).astype(np.float32)
        return np.clip(cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


# --------------------------------------------------------------------------
# anomaly kinds
# --------------------------------------------------------------------------

def _bezier_points(h: int, w: int, rng: np.random.Generator, n: int = 60) -> np.ndarray:
    """Quadratic Bezier stroke sampled as integer pixel coordinates."""
    margin = 0.12
    pt = lambda: np.array([rng.uniform(margin * w, (1 - margin) * w),
                           rng.uniform(margin * h, (1 - margin) * h)], np.float32)
    p0, p2 = pt(), pt()
    # keep strokes from being tiny
    while np.linalg.norm(p2 - p0) < 0.18 * min(h, w):
        p2 = pt()
    p1 = (p0 + p2) / 2 + np.array([rng.uniform(-0.2, 0.2) * w,
                                   rng.uniform(-0.2, 0.2) * h], np.float32)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return curve.astype(np.int32)


def make_scratch(shape: tuple[int, int], rng: np.random.Generator,
                 width: int | None = None) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    pts = _bezier_points(h, w, rng)
    width = int(width if width is not None else rng.integers(1, 4))
    cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, width, cv2.LINE_AA)
    return (m.astype(np.float32) / 255.0)


def make_blob(shape: tuple[int, int], rng: np.random.Generator,
              target_frac: float = 0.01) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.float32)
    area = target_frac * h * w
    rad = int(np.clip(np.sqrt(area / np.pi), 4, min(h, w) // 4))
    cx = int(rng.integers(rad + 1, max(w - rad - 1, rad + 2)))
    cy = int(rng.integers(rad + 1, max(h - rad - 1, rad + 2)))
    ax = int(rad * rng.uniform(0.7, 1.4))
    ay = int(rad * rng.uniform(0.7, 1.4))
    cv2.ellipse(m, (cx, cy), (max(ax, 2), max(ay, 2)),
                float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), sigmaX=max(rad / 4.0, 1.0))


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def paste_anomaly(img: np.ndarray, rng: np.random.Generator,
                  bank: TextureBank | None = None,
                  kind: str | None = None,
                  area_frac: float | None = None,
                  contrast: float | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, AnomalySpec]:
    """Return (image_with_anomaly, binary_mask, spec).

    The mask is the hard support of the anomaly; blending may be soft inside it.
    """
    x = np.asarray(img, np.float32).copy()
    if x.ndim == 2:
        x = np.repeat(x[:, :, None], 3, axis=2)
    h, w = x.shape[:2]

    kind = kind or str(rng.choice(ANOMALY_KINDS, p=[0.45, 0.35, 0.20]))

    if kind == "texture":
        frac = float(area_frac if area_frac is not None else rng.uniform(0.004, 0.030))
        soft = perlin_mask((h, w), rng, target_frac=frac)
        bank = bank or TextureBank()
        tex = bank.sample((h, w), rng)
        alpha = float(contrast if contrast is not None else rng.uniform(0.45, 0.95))
        m3 = soft[:, :, None]
        out = x * (1 - alpha * m3) + tex * (alpha * m3)
        hard = (soft > 0.5).astype(np.uint8)

    elif kind == "scratch":
        soft = make_scratch((h, w), rng)
        amp = float(contrast if contrast is not None else rng.uniform(0.12, 0.45))
        sign = -1.0 if rng.random() < 0.65 else 1.0      # scratches usually darker
        m3 = soft[:, :, None]
        out = x + sign * amp * m3
        hard = (soft > 0.25).astype(np.uint8)
        frac = float(hard.mean())

    else:  # blob
        frac = float(area_frac if area_frac is not None else rng.uniform(0.003, 0.020))
        soft = make_blob((h, w), rng, target_frac=frac)
        amp = float(contrast if contrast is not None else rng.uniform(0.06, 0.22))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        tint = np.array([rng.uniform(0.6, 1.0) for _ in range(3)], np.float32)
        out = x + sign * amp * soft[:, :, None] * tint[None, None, :]
        hard = (soft > 0.35).astype(np.uint8)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    spec = AnomalySpec(kind=kind,
                       area_frac=float(hard.mean()),
                       contrast=float(np.abs(out - x).max()))
    return out, hard, spec


def make_counterfactual_pair(clean: np.ndarray, rng: np.random.Generator,
                             bank: TextureBank | None = None, **kw):
    """(x_with_anomaly, x_clean, mask, spec) - the pair L_pres needs."""
    x_a, mask, spec = paste_anomaly(clean, rng, bank=bank, **kw)
    return x_a, np.asarray(clean, np.float32), mask, spec
