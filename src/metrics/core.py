"""Evaluation metrics.

The three proposed measures live here, plus the standard fidelity measures.
If these are wrong, every result in the project is wrong, so they are covered
by hand-constructed unit tests in tests/test_metrics.py.

Conventions: images are float32 in [0,1], shape (H,W,3) or (H,W).
Masks are binary uint8/bool, shape (H,W), 1 == anomaly.
"""
from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SKIMAGE = True
except Exception:                                    # pragma: no cover
    _HAVE_SKIMAGE = False

EPS = 1e-8


def _gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x.mean(axis=2) if x.ndim == 3 else x


def _as_bool(m: np.ndarray) -> np.ndarray:
    return np.asarray(m).astype(bool)


# --------------------------------------------------------------------------
# standard fidelity
# --------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR, optionally restricted to mask==True pixels."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = (a - b) ** 2
    if mask is not None:
        m = _as_bool(mask)
        if m.ndim == 2 and d.ndim == 3:
            m = m[:, :, None]
        m = np.broadcast_to(m, d.shape)
        if m.sum() == 0:
            return float("nan")
        mse = float(d[m].mean())
    else:
        mse = float(d.mean())
    if mse <= EPS:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if not _HAVE_SKIMAGE:                            # pragma: no cover
        return float("nan")
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.ndim == 3:
        return float(_ssim(a, b, channel_axis=2, data_range=1.0))
    return float(_ssim(a, b, data_range=1.0))


# --------------------------------------------------------------------------
# PROPOSED METRIC 1 - Defect Retention Ratio
# --------------------------------------------------------------------------

def defect_retention_ratio(restored_anom: np.ndarray,
                           restored_clean: np.ndarray,
                           clean_anom: np.ndarray,
                           clean_clean: np.ndarray,
                           mask: np.ndarray) -> float:
    """DRR - fraction of the defect residual that survives restoration.

        DRR = || (x~_a - x~_0) * M ||_1  /  || (x_a - x_0) * M ||_1

    where x~ denotes a restored image and x a clean one, subscript a means
    "with anomaly" and 0 means "without". The counterfactual pair must share
    the same degradation parameters AND the same noise seed.

    1.0 = defect perfectly preserved.  0.0 = defect completely erased.
    Values > 1 mean the restorer amplified the residual (possible, and worth
    reporting rather than clipping - it usually signals ringing artefacts).
    """
    m = _as_bool(mask)
    if m.sum() == 0:
        return float("nan")

    got = np.abs(_gray(restored_anom) - _gray(restored_clean))
    want = np.abs(_gray(clean_anom) - _gray(clean_clean))

    denom = float(want[m].sum())
    if denom <= EPS:
        return float("nan")
    return float(got[m].sum() / denom)


def relative_drr(drr_method: float, drr_identity: float) -> float:
    """DRR normalised by the no-op baseline.

    Raw DRR conflates two different losses: the degradation itself already
    attenuates a defect (blur spreads it, quantisation truncates it) before
    any restorer runs, so even the identity map scores below 1.0. Dividing by
    the identity score isolates the part attributable to RESTORATION, which is
    what the thesis is actually about.

    Report both: raw DRR says how much defect signal reaches the detector,
    relative DRR says how much the restorer destroyed.
    """
    if not np.isfinite(drr_identity) or abs(drr_identity) <= EPS:
        return float("nan")
    return float(drr_method / drr_identity)


# --------------------------------------------------------------------------
# PROPOSED METRIC 2 - Anomaly Contrast Gain
# --------------------------------------------------------------------------

def local_contrast(img: np.ndarray, mask: np.ndarray, ring: int = 9) -> float:
    """|mean_defect - mean_localbg| / std_localbg, using a dilated ring as bg."""
    import cv2
    g = _gray(img)
    m = _as_bool(mask)
    if m.sum() == 0:
        return float("nan")
    k = np.ones((ring, ring), np.uint8)
    dil = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)
    bg = dil & (~m)
    if bg.sum() < 8:
        bg = (~m)
    if bg.sum() == 0:
        return float("nan")
    sd = float(g[bg].std())
    return float(abs(g[m].mean() - g[bg].mean()) / (sd + EPS))


def anomaly_contrast_gain(restored: np.ndarray, degraded: np.ndarray,
                          mask: np.ndarray, ring: int = 9) -> float:
    """ACG = contrast_after / contrast_before. >1 means more separable."""
    c_after = local_contrast(restored, mask, ring)
    c_before = local_contrast(degraded, mask, ring)
    if not np.isfinite(c_before) or c_before <= EPS:
        return float("nan")
    return float(c_after / c_before)


# --------------------------------------------------------------------------
# PROPOSED METRIC 3 - Hallucinated Defect Rate
# --------------------------------------------------------------------------

def hallucinated_defect_rate(scores_restored_clean: np.ndarray,
                             threshold: float) -> float:
    """Fraction of DEFECT-FREE images pushed above threshold by restoration."""
    s = np.asarray(scores_restored_clean, np.float32).ravel()
    if s.size == 0:
        return float("nan")
    return float((s > threshold).mean())


# --------------------------------------------------------------------------
# PROPOSED METRIC 4 - Degradation Removal Ratio
# --------------------------------------------------------------------------

def degradation_removal_ratio(restored: np.ndarray, degraded: np.ndarray,
                              clean: np.ndarray,
                              mask: np.ndarray | None = None) -> float:
    """DRemR = 1 - ||x~ - x|| / ||y - x||, on NORMAL regions.

    1.0 = degradation perfectly removed, 0.0 = nothing done,
    negative = restoration moved further from clean than the degraded input.
    """
    r, d, c = _gray(restored), _gray(degraded), _gray(clean)
    if mask is not None:
        norm = ~_as_bool(mask)
    else:
        norm = np.ones(c.shape, bool)
    if norm.sum() == 0:
        return float("nan")
    num = float(np.abs(r[norm] - c[norm]).sum())
    den = float(np.abs(d[norm] - c[norm]).sum())
    if den <= EPS:
        return float("nan")
    return float(1.0 - num / den)


# --------------------------------------------------------------------------
# detection metrics
# --------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with correct tie handling. labels: 1 = anomalous."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def f1_max(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 and the threshold attaining it."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if y.sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return float(f1[k]), float(s_sorted[k])


def robustness_gap(auroc_clean: float, auroc_degraded: float) -> float:
    """The headline number. Lower is better; 0 means degradation cost nothing."""
    return float(auroc_clean - auroc_degraded)


def gap_closed(auroc_clean: float, auroc_degraded: float,
               auroc_method: float) -> float:
    """Fraction of the clean-to-degraded gap recovered by a method."""
    gap = auroc_clean - auroc_degraded
    if abs(gap) <= EPS:
        return float("nan")
    return float((auroc_method - auroc_degraded) / gap)
