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


def defect_residual_correlation(restored_anom: np.ndarray,
                                restored_clean: np.ndarray,
                                clean_anom: np.ndarray,
                                clean_clean: np.ndarray,
                                mask: np.ndarray) -> float:
    """Pearson correlation between the restored residual and the true
    residual, over mask pixels, both mean-centred within the mask - a
    SHAPE check alongside DRR's magnitude check.

    DRR (defect_retention_ratio) is ||got||/||want||, a ratio of
    magnitudes. Two residuals can have matched magnitude while pointing in
    completely different directions - e.g. ringing or hallucinated high-
    frequency content sitting at the defect's location, with similar
    energy to the true defect but no actual relationship to its shape -
    and DRR cannot tell that apart from genuine preservation. This can:
    it asks whether the restored residual's SHAPE tracks the true
    residual's shape, not just its size.

    1.0 = the restored residual is a positive scalar multiple of the true
    residual - genuine preservation. 0.0 = uncorrelated - restoration
    produced something with no relationship to the real defect (ringing,
    hallucinated texture). -1.0 = anti-correlated (a sign-flipped
    residual) - pathological, reported honestly rather than clamped.

    Read alongside DRR, not instead of it: a restorer that does nothing
    (identity) scores 1.0 on both. A restorer that erases the defect
    scores near 0 on DRR and is typically near 0 or undefined here too
    (little residual left to correlate against anything). The case this
    metric exists for is DRR > 0 (residual energy present) but this near
    0 (that energy isn't the defect's shape) - that combination is where
    DRR alone would have been misread as preservation.
    """
    m = _as_bool(mask)
    if m.sum() < 2:
        return float("nan")
    got = (_gray(restored_anom) - _gray(restored_clean))[m]
    want = (_gray(clean_anom) - _gray(clean_clean))[m]
    got = got - got.mean()
    want = want - want.mean()
    denom = float(np.sqrt(float((got ** 2).sum()) * float((want ** 2).sum())))
    if denom <= EPS:
        return float("nan")
    return float(float((got * want).sum()) / denom)


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

def degradation_magnitude(degraded: np.ndarray, clean: np.ndarray,
                          mask: np.ndarray | None = None) -> float:
    """||y - x||_1 on NORMAL regions - DRemR's denominator, exposed on its
    own because it sets DRemR's sensitivity to restoration error:
    d(DRemR)/d(restoration error) = -1/degradation_magnitude, so a FIXED
    absolute restoration-error change produces a LARGER DRemR swing on
    mildly-degraded examples (small magnitude here) than on severely
    degraded ones. That's an exact consequence of the ratio in
    degradation_removal_ratio's formula, not an empirical claim about any
    particular model or run."""
    d, c = _gray(degraded), _gray(clean)
    if mask is not None:
        norm = ~_as_bool(mask)
    else:
        norm = np.ones(c.shape, bool)
    if norm.sum() == 0:
        return float("nan")
    return float(np.abs(d[norm] - c[norm]).sum())


def degradation_removal_ratio(restored: np.ndarray, degraded: np.ndarray,
                              clean: np.ndarray,
                              mask: np.ndarray | None = None) -> float:
    """DRemR = 1 - ||x~ - x|| / ||y - x||, on NORMAL regions.

    1.0 = degradation perfectly removed, 0.0 = nothing done,
    negative = restoration moved further from clean than the degraded input.

    A ratio, not an absolute error: its denominator is degradation_magnitude(),
    which can be small (mild degradation) or large (severe degradation, esp.
    illumination) by construction - see that function's docstring for the
    exact sensitivity this implies. Measured on a real PCIM checkpoint
    (src/experiments/eval_pcim_holdout.py, 150 held-out MVTec examples) this
    is NOT simply "DRemR is noisier than PSNR, trust PSNR instead" - the
    distortion is directional and asymmetric:
      - small degradation_magnitude (severity 1, easy families): DRemR was
        CONSISTENTLY biased negative (mean -0.21 across the mildest third)
        despite excellent PSNR (33-40dB) on the same examples - a small,
        genuinely-present restoration error gets amplified by the small
        denominator into a DRemR that reads as "made it worse."
      - large degradation_magnitude (severity 3-5 illumination especially):
        DRemR sat near zero (mean +0.11) despite catastrophic PSNR (down to
        6-9dB) on the same examples - the same mechanism in reverse: a huge
        denominator deflates a large absolute error into a DRemR that reads
        as "did nothing," masking real failure.
      - the highest DRemR variance and the single worst outlier in that
        run (-4.03) came from the MIDDLE third of the degradation_magnitude
        range, not the smallest-denominator tail - so this ratio mechanism
        does not fully explain observed DRemR volatility; some of it is
        genuine, family/severity-dependent model behaviour.
    Net: always report PSNR alongside DRemR, and treat a DRemR near zero on
    severely-illumination-degraded examples as uninformative rather than
    reassuring - check PSNR there specifically.
    """
    r, d, c = _gray(restored), _gray(degraded), _gray(clean)
    if mask is not None:
        norm = ~_as_bool(mask)
    else:
        norm = np.ones(c.shape, bool)
    if norm.sum() == 0:
        return float("nan")
    num = float(np.abs(r[norm] - c[norm]).sum())
    den = degradation_magnitude(degraded, clean, mask)
    if not np.isfinite(den) or den <= EPS:
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
