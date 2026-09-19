"""DBDE - Defect-Blind Degradation Estimator.

Pure classical image processing. No learned weights, no GPU. Every statistic
used here is a *global aggregate*, which is what makes the estimator blind to
spatially sparse defects: perturbing 1% of pixels perturbs the estimate by
roughly 1%.

  noise sigma   <- flat-patch PCA (smallest eigenvalue of patch covariance)
  illumination  <- log-domain low-pass + robust low-order polynomial fit
  blur kernel   <- cepstral peak analysis + radially-averaged PSD
  compression   <- DCT coefficient histogram periodicity

This module is the strongest "this really is a DIP project" artefact, and it
is fully testable on CPU because we generate the ground truth ourselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-8


@dataclass
class DegradationEstimate:
    noise_sigma: float = 0.0
    illum_field: np.ndarray | None = None     # (H,W) multiplicative field
    illum_ev: float = 0.0
    blur_kind: str = "none"                   # none | defocus | motion
    blur_radius: float = 0.0
    blur_length: float = 0.0
    blur_angle: float = 0.0
    jpeg_qf: int = 100

    def kernel(self) -> np.ndarray:
        from src.degrade.simulator import defocus_kernel, motion_kernel
        if self.blur_kind == "defocus" and self.blur_radius > 0.3:
            return defocus_kernel(self.blur_radius)
        if self.blur_kind == "motion" and self.blur_length > 1.5:
            return motion_kernel(self.blur_length, self.blur_angle)
        k = np.zeros((3, 3), np.float32)
        k[1, 1] = 1.0
        return k


def _gray(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x.mean(axis=2) if x.ndim == 3 else x


# --------------------------------------------------------------------------
# 1. noise level  -  flat-patch PCA
# --------------------------------------------------------------------------

def estimate_noise_sigma(img: np.ndarray, patch: int = 8,
                         flat_quantile: float = 0.25,
                         max_patches: int = 20000) -> float:
    """Smallest eigenvalue of the covariance of the flattest patches.

    Defects are high-variance, so the flatness filter discards them. This is
    the core of the estimator's defect-blindness.
    """
    g = _gray(img)
    h, w = g.shape
    if h < patch * 2 or w < patch * 2:
        return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.0)))

    # dense-ish grid of patches
    stride = max(patch // 2, 1)
    ys = np.arange(0, h - patch + 1, stride)
    xs = np.arange(0, w - patch + 1, stride)
    if len(ys) * len(xs) > max_patches:
        step = int(np.ceil(np.sqrt(len(ys) * len(xs) / max_patches)))
        ys, xs = ys[::step], xs[::step]

    patches = np.stack([g[y:y + patch, x:x + patch].ravel()
                        for y in ys for x in xs], axis=0)
    if patches.shape[0] < 16:
        return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.0)))

    var = patches.var(axis=1)
    keep = var <= np.quantile(var, flat_quantile)
    flat = patches[keep]
    if flat.shape[0] < 16:
        flat = patches

    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0.0
    # smallest eigenvalues are dominated by the noise floor
    k = max(int(0.10 * ev.size), 1)
    return float(np.sqrt(max(np.median(ev[:k]), 0.0)))


# --------------------------------------------------------------------------
# 2. illumination  -  log-domain low-pass + robust polynomial fit
# --------------------------------------------------------------------------

def _poly_design(h: int, w: int, degree: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(w - 1, 1)) * 2 - 1
    yn = (yy / max(h - 1, 1)) * 2 - 1
    cols = []
    for dy in range(degree + 1):
        for dx in range(degree + 1 - dy):
            cols.append((xn ** dx) * (yn ** dy))
    return np.stack([c.ravel() for c in cols], axis=1)


def estimate_illumination(img: np.ndarray, degree: int = 3,
                          huber_iters: int = 6,
                          prefilter: int = 5) -> tuple[np.ndarray, float]:
    """Return (normalised multiplicative field L, relative EV of that field).

    The fit is performed directly on the log image with iteratively reweighted
    (Huber) least squares. An earlier version low-pass filtered first, which
    was a mistake: a Gaussian smears a localised defect into a broad smooth
    bump that the robust weights can no longer distinguish from genuine
    illumination, so the defect gets absorbed into L. Fitting the raw log
    image instead lets Huber see defect pixels as the sharp outliers they are
    and downweight them, while the low polynomial degree still prevents the
    surface from bending to fit local structure. A small median prefilter
    removes sensor noise without spreading outliers.

    L is normalised to unit geometric mean. The absolute exposure level is
    deliberately NOT included: from a single image, scene albedo and
    illumination gain are not separable, so any global factor here would be
    arbitrary. The returned EV describes the spread of the field only.
    """
    g = _gray(img)
    h, w = g.shape

    if prefilter and prefilter >= 3:
        k = int(prefilter) | 1
        g = cv2.medianBlur((np.clip(g, 0, 1) * 255).astype(np.uint8), k).astype(np.float32) / 255.0

    logg = np.log(np.clip(g, 1e-3, None))

    A = _poly_design(h, w, degree)
    b = logg.ravel()
    wts = np.ones_like(b)
    coef = np.zeros(A.shape[1], np.float32)
    for _ in range(max(huber_iters, 1)):
        Aw = A * wts[:, None]
        coef, *_ = np.linalg.lstsq(Aw, b * wts, rcond=None)
        r = b - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + EPS
        delta = 1.0 * s
        wts = np.where(np.abs(r) <= delta, 1.0, delta / (np.abs(r) + EPS))

    fit = (A @ coef).reshape(h, w)
    fit = fit - fit.mean()                      # unit geometric mean
    L = np.clip(np.exp(fit), 0.2, 5.0).astype(np.float32)
    ev = float((fit.max() - fit.min()) / np.log(2.0))
    return L, ev


def correct_illumination(img: np.ndarray, L: np.ndarray,
                         ev: float = 0.0, clamp: float = 4.0,
                         match_mean: float | None = None) -> np.ndarray:
    """Divide out the spatial illumination field.

    Only the *spatial* variation is corrected. `ev` is accepted for interface
    compatibility but intentionally ignored, because the global exposure level
    cannot be recovered from one image (see estimate_illumination). If a
    reference brightness is known - and it is, since we hold clean normals for
    every class - pass it as `match_mean` to restore the global level too.
    """
    x = np.asarray(img, np.float32)
    field = np.clip(L, 1.0 / clamp, clamp)
    if x.ndim == 3:
        field = field[:, :, None]
    out = x / field

    if match_mean is not None:
        m = float(out.mean())
        if m > EPS:
            out = out * (float(match_mean) / m)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# 3. blur  -  cepstrum + radially averaged PSD
# --------------------------------------------------------------------------

def _windowed_fft(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    win = np.outer(wy, wx)
    return np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))


def radial_psd(img: np.ndarray, nbins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density. Returns (freq, power)."""
    g = _gray(img)
    F = _windowed_fft(g)
    P = (np.abs(F) ** 2).astype(np.float64)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    bins = np.linspace(0, rmax, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < nbins)
    power = np.bincount(idx[valid], weights=P.ravel()[valid], minlength=nbins)
    count = np.bincount(idx[valid], minlength=nbins).astype(np.float64)
    power = power / np.maximum(count, 1)
    freq = (bins[:-1] + bins[1:]) / 2 / max(rmax, 1)
    return freq.astype(np.float32), power.astype(np.float32)


def cepstrum(img: np.ndarray, signed: bool = True) -> np.ndarray:
    """Cepstrum of the log-magnitude spectrum.

    The SIGN matters: convolution by a finite-support kernel puts a
    characteristic *negative* dip at the lag equal to the kernel extent.
    Taking the magnitude (as is often done for display) destroys exactly the
    feature we need, so `signed=True` is the default here.
    """
    g = _gray(img)
    F = _windowed_fft(g)
    logmag = np.log(np.abs(F) + 1.0)
    c = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(logmag)))
    c = np.real(c) if signed else np.abs(c)
    return c.astype(np.float32)


def _motion_cepstral_peak(img: np.ndarray, max_len: int = 24,
                          min_len: int = 6) -> dict[str, float]:
    """The actual cepstral peak search. Returns length/angle/prominence/harm
    SEPARATELY (not just the blended confidence estimate_motion_blur()
    exposes) so the blur-decision logic can gate on prominence and harmonic
    confirmation independently rather than through one blended score a
    strong prominence alone can carry over threshold.

    A linear motion kernel of length L makes |K(f)| vanish periodically, which
    appears in the cepstrum as a negative dip at lag L and weaker echoes at
    2L, 3L. We locate the strongest dip outside a DC exclusion disc and then
    *confirm* it by checking for the first harmonic. Lags below `min_len` are
    not considered, since the DC lobe dominates there and a sub-6px motion
    blur is in any case negligible.

    Caution (found the hard way): periodic real-world texture - carpet weave,
    screw threads, bottle rims - produces cepstral dips indistinguishable from
    a genuine blur signature, AND periodic texture has harmonics too, so
    harmonic confirmation alone does not rule it out either. Neither
    prominence nor harm from this function should be trusted alone as a
    "there is blur" decision - see _detect_blur_reference/_detect_blur_blind,
    which additionally require the claimed kernel to explain the image's
    actual high-frequency energy loss, something periodic texture cannot
    mimic (reference mode) or is much harder to fake (blind mode).
    """
    c = cepstrum(img, signed=True)
    h, w = c.shape
    cy, cx = h // 2, w // 2

    win = int(min(max_len, cy - 2, cx - 2))
    if win < min_len + 2:
        return dict(length=0.0, angle=0.0, prominence=0.0, harm=0.0)

    yy, xx = np.mgrid[-win:win + 1, -win:win + 1]
    rad = np.sqrt(yy ** 2 + xx ** 2)
    patch = c[cy - win:cy + win + 1, cx - win:cx + win + 1].copy()

    valid = (rad >= min_len) & (rad <= win)
    if not valid.any():
        return dict(length=0.0, angle=0.0, prominence=0.0, harm=0.0)

    masked = np.where(valid, patch, np.inf)
    k = int(np.argmin(masked))
    py, px = np.unravel_index(k, masked.shape)
    dy, dx = py - win, px - win
    length = float(np.hypot(dy, dx))
    angle = float(np.arctan2(dy, dx) % np.pi)
    dip = float(patch[py, px])

    bg = patch[valid]
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med))) + EPS
    prominence = (med - dip) / (6.0 * mad)

    # harmonic confirmation at 2x the lag, if it fits inside the window
    harm = 0.0
    hy, hx = int(round(2 * dy)), int(round(2 * dx))
    if abs(hy) <= win and abs(hx) <= win:
        neigh = patch[max(hy + win - 1, 0):hy + win + 2,
                      max(hx + win - 1, 0):hx + win + 2]
        if neigh.size:
            harm = float(np.clip((med - neigh.min()) / (6.0 * mad), 0.0, 1.0))

    return dict(length=length, angle=angle, prominence=float(np.clip(prominence, 0, 1)),
               harm=harm)


def estimate_motion_blur(img: np.ndarray, max_len: int = 24,
                         min_len: int = 6) -> tuple[float, float, float]:
    """Return (length_px, angle_rad, confidence) from the signed cepstrum.

    Thin wrapper around _motion_cepstral_peak() for backward compatibility -
    `confidence` blends prominence and harmonic confirmation into one score.
    The blur-decision logic in estimate() does NOT use this blended score to
    decide whether blur is present (see _motion_cepstral_peak's docstring for
    why); it calls _motion_cepstral_peak directly instead. This function
    still stands alone for callers that just want a cepstral peak estimate.
    """
    d = _motion_cepstral_peak(img, max_len, min_len)
    conf = float(np.clip(0.65 * d["prominence"] + 0.35 * d["harm"], 0.0, 1.0))
    return d["length"], d["angle"], conf


def _radial_mtf(kernel: np.ndarray, freq: np.ndarray, n: int = 256) -> np.ndarray:
    """Radial profile of |FFT(kernel)| sampled at the given normalised freqs."""
    K = np.abs(np.fft.fftshift(np.fft.fft2(kernel, s=(n, n))))
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c
    nb = len(freq)
    edges = np.linspace(0, 1, nb + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nb)
    tot = np.bincount(idx[ok], weights=K.ravel()[ok], minlength=nb)
    cnt = np.bincount(idx[ok], minlength=nb).astype(np.float64)
    prof = tot / np.maximum(cnt, 1)
    return np.maximum(prof, EPS)


def reference_psd(images: list[np.ndarray], nbins: int = 128) -> np.ndarray:
    """Mean log-PSD over a set of CLEAN images of one class.

    Industrial inspection has a structural advantage over generic blind
    restoration: defect-free reference images of the exact part are always
    available, because that is what the detector was fitted on. Storing their
    average spectrum turns blind blur estimation into a far easier reference-
    based problem. Compute this once per category and cache it.
    """
    acc = None
    for im in images:
        _, pw = radial_psd(im, nbins=nbins)
        lp = np.log(np.maximum(pw, EPS))
        acc = lp if acc is None else acc + lp
    if acc is None:
        raise ValueError("reference_psd needs at least one image")
    return (acc / len(images)).astype(np.float32)


def _signal_band(freq: np.ndarray, logp: np.ndarray) -> slice:
    """The frequency band used for curve-fitting: excludes the DC/near-DC
    region (dominated by content, not blur) and the noise floor at the very
    top of the spectrum. Shared by defocus radius fitting and the
    reference-based high-frequency-deficit blur gate, so both operate over
    the same physically-meaningful band."""
    lo = max(int(0.06 * len(freq)), 2)
    floor = float(np.median(logp[-8:]))
    above = np.where(logp > floor + 0.35)[0]
    hi = int(above[-1]) + 1 if above.size else int(0.80 * len(freq))
    hi = int(np.clip(hi, lo + 12, len(freq)))
    return slice(lo, hi)


def estimate_defocus_radius(img: np.ndarray, max_radius: float = 8.0,
                            step: float = 0.25,
                            ref_logpsd: np.ndarray | None = None
                            ) -> tuple[float, float]:
    """Estimate disc radius by fitting the PSD roll-off.

    Two modes:

    reference-based (`ref_logpsd` given) - the observed log-PSD minus the
    clean class reference is exactly 2 log|K(f)|, so the radius follows from a
    direct curve match. This is accurate and is the mode used in the pipeline,
    since clean normals are always on hand.

    blind (no reference) - the image's own spectrum is unknown, so it is
    absorbed by a free affine term in log-frequency (a power-law prior). This
    is noticeably weaker at large radii, where little signal survives above
    the noise floor, and is kept only as a fallback.
    """
    from src.degrade.simulator import defocus_kernel

    freq, power = radial_psd(img)
    logp = np.log(np.maximum(power, EPS))
    band = _signal_band(freq, logp)

    radii = np.arange(0.0, max_radius + step, step)

    if ref_logpsd is not None:
        ratio = logp - np.asarray(ref_logpsd, np.float32)
        errs = []
        for r in radii:
            model = (np.zeros_like(freq) if r < step
                     else 2.0 * np.log(_radial_mtf(defocus_kernel(float(r)), freq)))
            errs.append(float(np.mean((ratio[band] - model[band]) ** 2)))
        errs = np.asarray(errs)
    else:
        f = np.maximum(freq[band], 1e-3)
        y = logp[band]
        base = np.stack([np.ones_like(f), np.log(f)], axis=1)
        errs = []
        for r in radii:
            model = (np.zeros_like(freq) if r < step
                     else 2.0 * np.log(_radial_mtf(defocus_kernel(float(r)), freq)))
            A = np.concatenate([base, model[band][:, None]], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            err = float(np.mean(resid ** 2)) + (10.0 if coef[2] < 0.25 else 0.0)
            errs.append(err)
        errs = np.asarray(errs)

    best = int(np.argmin(errs))
    best_r = float(radii[best])
    spread = float(errs.max() - errs.min())
    conf = float(np.clip(spread / (errs.mean() + EPS), 0.0, 1.0))
    return best_r, conf


# --------------------------------------------------------------------------
# blur presence/type decision
# --------------------------------------------------------------------------
#
# estimate_motion_blur()'s cepstral peak and blind estimate_defocus_radius()
# both answer "if there IS a kernel, what does it look like" - neither is a
# reliable answer to "is there a kernel at all." Real photographs have
# periodic structure everywhere (carpet weave, screw threads, bottle rims)
# that produces cepstral dips and harmonics indistinguishable from genuine
# motion blur, and a free-affine-term curve fit can find a small nonzero
# defocus radius that "improves" the fit on pure noise. Found by running the
# blind decision below (motion checked first, gated only on the blended
# cepstral confidence) against real MVTec images across every degradation
# family: it classified every single eval example as "motion", including
# images degraded by illumination/noise/jpeg alone, which have no blur at
# all. That is a design flaw in the decision procedure, not a threshold that
# needed nudging - see the two functions below.

# Known residual, not swept under the rug: strong illumination changes
# (severity 4-5, under-exposure specifically) still produce a real,
# non-trivial false-positive rate here (roughly 10-65% depending on
# severity/direction, measured - see test_reference_blur_detection_on_
# severe_illumination_is_bounded_not_zero). This is physically principled,
# not a detector defect: severe under-exposure genuinely destroys high-
# frequency content (quantisation and read noise swamp fine detail at low
# signal), so the detector correctly observes an HF deficit and
# misattributes its cause - a known hard case in blind deconvolution, not
# a bug to chase with another threshold. The improve-margin distributions
# for genuine severity-2 motion (0.49-0.95) and spurious under-exposure
# "motion" (0.57-0.74) genuinely overlap; tightening further would trade
# away real severity-2 motion sensitivity to suppress it.
#
# FUTURE WORK (not implemented - flagging only): illumination is already
# estimated and corrected elsewhere in this module before blur matters to
# anything downstream. Running blur detection on the illumination-
# corrected image (rather than the raw one) should shrink exactly the
# deficit that's fooling it here, since much of that deficit is the
# exposure change itself, not a separate blur. Worth trying before
# reaching for another threshold if this residual ever needs to shrink
# further.

def _highfreq_deficit(logp: np.ndarray, band: slice, ref_logpsd: np.ndarray) -> float:
    """Mean high-frequency log-power lost relative to the category's clean
    reference. Blur suppresses high-frequency energy; periodic texture does
    not, because the SAME texture is present in the reference too. Positive
    = energy lost (blur-consistent). This is the one test in this file that
    periodic real-world structure cannot fake, because it's a difference
    against an image of the same physical content, not a property of this
    image alone."""
    ratio = logp - np.asarray(ref_logpsd, np.float32)
    return float(-np.mean(ratio[band]))


def _detect_blur_reference(img: np.ndarray, ref_logpsd: np.ndarray,
                           deficit_margin: float = 1.0,
                           fit_improvement_margin: float = 0.3,
                           min_radius: float = 1.5,
                           min_length: float = 5.0,
                           max_radius: float = 8.0
                           ) -> tuple[str, float, float, float]:
    """Reference-based blur presence/type decision - the primary path
    whenever a category reference is available.

    Three gates, all required:
      1. deficit_margin - the image must actually be missing high-frequency
         energy relative to its own class's clean reference, by a real
         margin (in log-power / nats). This is the physical test periodic
         texture cannot mimic.
      2. fit_improvement_margin - among the candidate kernel families
         (defocus, motion), the winning one must substantially reduce the
         fit residual against the deficit curve versus explaining nothing.
         A deficit can be real without matching any of our kernel shapes
         (e.g. sensor-level softening, or - found empirically - highlight
         clipping from strong exposure changes, which produces a broadband
         high-frequency deficit that a very small defocus radius fits well
         enough to pass a loose version of this gate).
      3. min_radius / min_length - the winning kernel must be non-trivially
         sized. Found empirically: exposure-clipping's spurious "best fit"
         defocus radius clustered right around 1.0px, indistinguishable in
         size from a genuine severity-1 defocus blur (DEFOCUS_RADIUS[0] is
         also 1.0px) - the deficit-shape gate alone doesn't separate them,
         because at that size they really do look similar. Requiring a
         larger minimum trades away severity-1 sensitivity (severity-1
         defocus, true radius 1.0px, now goes undetected) for rejecting
         this specific false-positive mode; severity 2+ (radius >=2px) is
         unaffected. Motion's floor mirrors _motion_cepstral_peak's own
         min_len=6px cepstral search floor - lengths below that were never
         reliably estimated in the first place, so this costs nothing
         beyond what min_len already cost.

    Returns (blur_kind, blur_radius, blur_length, blur_angle).
    """
    from src.degrade.simulator import defocus_kernel, motion_kernel

    freq, power = radial_psd(img)
    logp = np.log(np.maximum(power, EPS))
    band = _signal_band(freq, logp)
    ratio = logp - np.asarray(ref_logpsd, np.float32)

    deficit = _highfreq_deficit(logp, band, ref_logpsd)
    if deficit < deficit_margin:
        return "none", 0.0, 0.0, 0.0

    mse_none = float(np.mean(ratio[band] ** 2))

    r_best, _ = estimate_defocus_radius(img, max_radius=max_radius, ref_logpsd=ref_logpsd)
    if r_best >= min_radius:
        model = 2.0 * np.log(_radial_mtf(defocus_kernel(r_best), freq))
        mse_defocus = float(np.mean((ratio[band] - model[band]) ** 2))
    else:
        r_best, mse_defocus = 0.0, mse_none

    peak = _motion_cepstral_peak(img)
    if peak["length"] >= min_length:
        model = 2.0 * np.log(_radial_mtf(motion_kernel(peak["length"], peak["angle"]), freq))
        mse_motion = float(np.mean((ratio[band] - model[band]) ** 2))
    else:
        mse_motion = mse_none

    best_kind, best_mse = "none", mse_none
    if mse_defocus < best_mse:
        best_kind, best_mse = "defocus", mse_defocus
    if mse_motion < best_mse:
        best_kind, best_mse = "motion", mse_motion

    improvement = (mse_none - best_mse) / (mse_none + EPS)
    if best_kind == "none" or improvement < fit_improvement_margin:
        return "none", 0.0, 0.0, 0.0
    if best_kind == "defocus":
        return "defocus", r_best, 0.0, 0.0
    return "motion", 0.0, peak["length"], peak["angle"]


def _detect_blur_blind(img: np.ndarray,
                       prominence_floor: float = 0.75,
                       harm_floor: float = 0.5,
                       shape_coef_floor: float = 0.4,
                       defocus_min_radius: float = 1.5,
                       max_radius: float = 8.0) -> tuple[str, float, float, float]:
    """Blind fallback for when no category reference exists yet. "No blur"
    is the default outcome, not a rare one: motion is only accepted if the
    cepstral prominence AND the harmonic confirmation AND a shape-consistency
    check against the image's own spectrum all agree; if any one disagrees,
    report no blur. All three thresholds are deliberately far stricter than
    the reference-mode gates, since without a reference this whole
    procedure is guessing at what "no blur" would have looked like for this
    specific image rather than measuring it directly.
    """
    from src.degrade.simulator import motion_kernel

    freq, power = radial_psd(img)
    logp = np.log(np.maximum(power, EPS))
    band = _signal_band(freq, logp)

    peak = _motion_cepstral_peak(img)
    if (peak["length"] >= 2.0 and peak["prominence"] >= prominence_floor
            and peak["harm"] >= harm_floor):
        coef2, improvement = _blind_shape_fit(freq, logp, band,
                                              motion_kernel(peak["length"], peak["angle"]))
        if coef2 >= shape_coef_floor and improvement > 0:
            return "motion", 0.0, peak["length"], peak["angle"]

    d_rad, d_conf = estimate_defocus_radius(img, max_radius=max_radius, ref_logpsd=None)
    if d_rad >= defocus_min_radius and d_conf >= prominence_floor:
        return "defocus", d_rad, 0.0, 0.0

    return "none", 0.0, 0.0, 0.0


def _blind_shape_fit(freq: np.ndarray, logp: np.ndarray, band: slice,
                     kernel: np.ndarray) -> tuple[float, float]:
    """Blind (no-reference) shape-consistency check: fit the image's own
    log-PSD with a free affine term (absorbing the unknown natural-image
    spectral slope) PLUS the candidate kernel's frequency response, and
    return (kernel coefficient, fractional improvement over the affine-only
    fit). Mirrors the free-affine-term regression estimate_defocus_radius
    already uses in blind mode, generalised to any kernel - a real kernel
    should explain a meaningful share of the high-frequency roll-off beyond
    what a generic power-law prior already explains; periodic texture
    produces a strong cepstral peak but does not reshape the broadband
    roll-off this way.
    """
    f = np.maximum(freq[band], 1e-3)
    y = logp[band]
    base = np.stack([np.ones_like(f), np.log(f)], axis=1)
    model = 2.0 * np.log(_radial_mtf(kernel, freq))

    coef0, *_ = np.linalg.lstsq(base, y, rcond=None)
    mse_base = float(np.mean((y - base @ coef0) ** 2))

    A = np.concatenate([base, model[band][:, None]], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    mse_full = float(np.mean((y - A @ coef) ** 2))

    improvement = (mse_base - mse_full) / (mse_base + EPS)
    return float(coef[2]), improvement


# --------------------------------------------------------------------------
# 4. compression  -  DCT histogram periodicity
# --------------------------------------------------------------------------

def estimate_jpeg_qf(img: np.ndarray, min_excess: float = 0.18) -> int:
    """JPEG quality estimate from blockiness at the 8x8 grid.

    Compares the mean gradient on the block boundary against the mean gradient
    just inside the block (offsets 3 and 4), rather than against all non-grid
    positions: any periodicity the image itself contains at other offsets then
    cancels out. Requires a clear margin before reporting compression at all,
    so mildly self-similar textures are not mistaken for JPEG artefacts.
    """
    g = _gray(img) * 255.0
    h, w = g.shape
    if h < 32 or w < 32:
        return 100

    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))

    def ratio(d: np.ndarray, axis: int) -> float:
        n = d.shape[axis]
        on = np.take(d, np.arange(7, n, 8), axis=axis)
        # mid-block reference offsets, far from the boundary
        ref_idx = np.concatenate([np.arange(3, n, 8), np.arange(4, n, 8)])
        ref_idx = ref_idx[ref_idx < n]
        if on.size == 0 or ref_idx.size == 0:
            return 1.0
        off = np.take(d, ref_idx, axis=axis)
        return float(on.mean() / (off.mean() + EPS))

    r = 0.5 * (ratio(dv, 1) + ratio(dh, 0))
    excess = r - 1.0
    if excess < min_excess:
        return 100
    qf = int(np.clip(100.0 - 95.0 * (excess - min_excess) - 5.0, 20, 99))
    return qf


# --------------------------------------------------------------------------
# top-level estimator
# --------------------------------------------------------------------------

def estimate(img: np.ndarray,
             poly_degree: int = 3,
             ref_logpsd: np.ndarray | None = None,
             deficit_margin: float = 1.0,
             fit_improvement_margin: float = 0.3,
             prominence_floor: float = 0.75,
             harm_floor: float = 0.5,
             shape_coef_floor: float = 0.4) -> DegradationEstimate:
    """Run the full defect-blind estimation chain on one image.

    Pass `ref_logpsd` (from reference_psd() over that category's clean
    training normals) whenever it is available - it is the PRIMARY blur
    detection path, not just an accuracy boost: blur suppresses high-
    frequency energy relative to the reference, a direct physical test that
    periodic real-world texture (carpet weave, screw threads, bottle rims)
    cannot mimic, because the same texture is present in the reference too.
    Without a reference, blur detection falls back to a much more
    conservative blind decision where "no blur" is the default outcome, not
    a rare one - see _detect_blur_reference / _detect_blur_blind.
    """
    est = DegradationEstimate()

    est.noise_sigma = estimate_noise_sigma(img)
    est.illum_field, est.illum_ev = estimate_illumination(img, degree=poly_degree)
    est.jpeg_qf = estimate_jpeg_qf(img)

    if ref_logpsd is not None:
        kind, radius, length, angle = _detect_blur_reference(
            img, ref_logpsd, deficit_margin=deficit_margin,
            fit_improvement_margin=fit_improvement_margin)
    else:
        kind, radius, length, angle = _detect_blur_blind(
            img, prominence_floor=prominence_floor, harm_floor=harm_floor,
            shape_coef_floor=shape_coef_floor)

    est.blur_kind = kind
    est.blur_radius = radius
    est.blur_length = length
    est.blur_angle = angle

    return est


def parameter_error(est: DegradationEstimate, truth) -> dict[str, float]:
    """Absolute errors against ground-truth DegradationParams."""
    return {
        "noise_sigma_err": abs(est.noise_sigma - truth.noise_read),
        "blur_radius_err": abs(est.blur_radius - truth.blur_radius),
        "blur_length_err": abs(est.blur_length - truth.blur_length),
        "blur_kind_correct": float(est.blur_kind == truth.blur_kind),
        "jpeg_qf_err": abs(est.jpeg_qf - truth.jpeg_qf),
    }
