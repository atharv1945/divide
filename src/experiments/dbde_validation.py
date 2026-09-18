"""DBDE validation figures - pure classical DIP, no GPU, report-ready.

Three figures, each answering a specific question about the defect-blind
degradation estimator:

  defect_blindness.png      - does a defect perturb the estimate? (should be
                               roughly flat vs. defect area fraction - this is
                               the strongest evidence this is a DIP project,
                               not a deep-learning project wearing a DIP
                               costume, since nothing here is learned)
  parameter_accuracy.png    - estimated vs. true blur radius / motion length
                               / noise sigma, across all severities, with
                               error bars over multiple real images
  reference_vs_blind.png    - how much does having clean-normals reference
                               PSDs actually buy blur estimation? quantified,
                               not asserted

Run on real MVTec (falls back to synthetic smoke data with --smoke, but the
report wants the real-data version):

    python -m src.experiments.dbde_validation --categories carpet bottle screw
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.data.mvtec import DEFAULT_CATEGORIES, load_train_normals, dataset_available
from src.dbde.estimator import (
    estimate, estimate_defocus_radius, estimate_motion_blur, estimate_noise_sigma,
    reference_psd,
)
from src.degrade.anomaly import paste_anomaly
from src.degrade.simulator import (
    DEFOCUS_RADIUS, MOTION_LENGTH, NOISE_READ, SEVERITIES,
    DegradationParams, apply_degradation,
)
from src.utils.paths import figures_dir


def _pool_images(categories: list[str], n_per_category: int, size: int,
                 smoke: bool, seed: int) -> list[np.ndarray]:
    imgs = []
    for cat in categories:
        imgs += load_train_normals(cat, size=size, limit=n_per_category, smoke=smoke)
    if not imgs:
        raise RuntimeError("no images loaded - check DIVIDE_DATA_ROOT or pass --smoke")
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    return imgs


# --------------------------------------------------------------------------
# 1. defect blindness vs. area fraction
# --------------------------------------------------------------------------

def defect_blindness_data(images: list[np.ndarray], area_fracs: list[float],
                          seed: int) -> dict[str, list[float]]:
    """For each area fraction, mean |estimate(with defect) - estimate(without)|
    across images and a handful of degradation families, per estimated
    parameter. A roughly-flat curve is the defect-blindness claim; a rising
    one means defects are leaking into the estimate."""
    rng = np.random.default_rng(seed)
    families = [
        ("noise", lambda p: DegradationParams(family="noise", severity=3,
                                              noise_read=NOISE_READ[2] / 255),
         lambda e: e.noise_sigma, "noise_sigma"),
        ("defocus", lambda p: DegradationParams(family="defocus", severity=3,
                                                blur_kind="defocus",
                                                blur_radius=DEFOCUS_RADIUS[2]),
         lambda e: e.blur_radius, "blur_radius"),
        ("motion", lambda p: DegradationParams(family="motion", severity=3,
                                               blur_kind="motion",
                                               blur_length=MOTION_LENGTH[2], blur_angle=0.3),
         lambda e: e.blur_length, "blur_length"),
    ]

    out = {name: [] for name, *_ in families}
    for frac in area_fracs:
        per_family_errs = {name: [] for name, *_ in families}
        for img in images:
            xa, mask, _ = paste_anomaly(img, rng, kind="texture", area_frac=frac)
            if mask.sum() == 0:
                continue
            for name, make_params, extract, _ in families:
                p = make_params(None)
                y_clean = apply_degradation(img, p, np.random.default_rng(seed))
                y_anom = apply_degradation(xa, p, np.random.default_rng(seed))
                e0 = extract(estimate(y_clean))
                e1 = extract(estimate(y_anom))
                per_family_errs[name].append(abs(e1 - e0))
        for name, *_ in families:
            errs = per_family_errs[name]
            out[name].append(float(np.mean(errs)) if errs else float("nan"))
    return out


def make_defect_blindness_figure(images: list[np.ndarray], out_path, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area_fracs = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    data = defect_blindness_data(images, area_fracs, seed)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    titles = dict(noise_sigma="noise sigma drift", blur_radius="defocus radius drift",
                 blur_length="motion length drift")
    for ax, (name, ys) in zip(axes, data.items()):
        ax.plot(area_fracs, ys, marker="o")
        ax.set_xlabel("defect area fraction")
        ax.set_ylabel("|estimate(with) - estimate(without)|")
        ax.set_title(titles.get(name, name))
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    fig.suptitle("DBDE defect-blindness: estimation drift vs. defect area "
                "(flat = defect-blind)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return data


# --------------------------------------------------------------------------
# 2. parameter accuracy across severities
# --------------------------------------------------------------------------

def parameter_accuracy_data(images: list[np.ndarray], seed: int) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    results = {"blur_radius": {"true": [], "mean": [], "std": []},
              "blur_length": {"true": [], "mean": [], "std": []},
              "noise_sigma": {"true": [], "mean": [], "std": []}}

    for i, sev in enumerate(SEVERITIES):
        r_true = DEFOCUS_RADIUS[i]
        ests = []
        for img in images:
            p = DegradationParams(family="defocus", severity=sev, blur_kind="defocus",
                                  blur_radius=r_true)
            y = apply_degradation(img, p, np.random.default_rng(seed + i))
            ests.append(estimate_defocus_radius(y)[0])
        results["blur_radius"]["true"].append(r_true)
        results["blur_radius"]["mean"].append(float(np.mean(ests)))
        results["blur_radius"]["std"].append(float(np.std(ests)))

        len_true = MOTION_LENGTH[i]
        ests = []
        for img in images:
            p = DegradationParams(family="motion", severity=sev, blur_kind="motion",
                                  blur_length=len_true, blur_angle=0.4)
            y = apply_degradation(img, p, np.random.default_rng(seed + i))
            est_len, _, conf = estimate_motion_blur(y)
            if conf > 0.0:
                ests.append(est_len)
        results["blur_length"]["true"].append(len_true)
        results["blur_length"]["mean"].append(float(np.mean(ests)) if ests else float("nan"))
        results["blur_length"]["std"].append(float(np.std(ests)) if ests else float("nan"))

        sigma_true = NOISE_READ[i] / 255.0
        ests = []
        for img in images:
            p = DegradationParams(family="noise", severity=sev, noise_read=sigma_true)
            y = apply_degradation(img, p, np.random.default_rng(seed + i))
            ests.append(estimate_noise_sigma(y))
        results["noise_sigma"]["true"].append(sigma_true)
        results["noise_sigma"]["mean"].append(float(np.mean(ests)))
        results["noise_sigma"]["std"].append(float(np.std(ests)))

    return results


def make_parameter_accuracy_figure(images: list[np.ndarray], out_path, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = parameter_accuracy_data(images, seed)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    labels = dict(blur_radius="defocus radius (px)", blur_length="motion length (px)",
                 noise_sigma="noise sigma")
    for ax, (name, d) in zip(axes, data.items()):
        true = np.array(d["true"])
        mean = np.array(d["mean"])
        std = np.array(d["std"])
        ax.errorbar(true, mean, yerr=std, fmt="o-", capsize=3)
        lo, hi = min(true.min(), np.nanmin(mean)), max(true.max(), np.nanmax(mean))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="perfect")
        ax.set_xlabel(f"true {labels[name]}")
        ax.set_ylabel(f"estimated {labels[name]}")
        ax.set_title(labels[name])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("DBDE parameter accuracy across severities (error bars = std over images)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return data


# --------------------------------------------------------------------------
# 3. reference vs. blind blur estimation
# --------------------------------------------------------------------------

def reference_vs_blind_data(images: list[np.ndarray], seed: int) -> dict[str, list[float]]:
    if len(images) < 4:
        raise ValueError("need at least 4 images: some held out for the reference PSD")
    ref_images, test_images = images[:3], images[3:]
    ref_logpsd = reference_psd(ref_images)

    radii = DEFOCUS_RADIUS
    ref_errs, blind_errs = [], []
    for i, r_true in enumerate(radii):
        p = DegradationParams(family="defocus", severity=i + 1, blur_kind="defocus",
                              blur_radius=r_true)
        ref_es, blind_es = [], []
        for img in test_images:
            y = apply_degradation(img, p, np.random.default_rng(seed + i))
            ref_es.append(estimate_defocus_radius(y, ref_logpsd=ref_logpsd)[0])
            blind_es.append(estimate_defocus_radius(y)[0])
        ref_errs.append(float(np.mean(np.abs(np.array(ref_es) - r_true))))
        blind_errs.append(float(np.mean(np.abs(np.array(blind_es) - r_true))))

    return dict(radii=radii, reference_mae=ref_errs, blind_mae=blind_errs)


def make_reference_vs_blind_figure(images: list[np.ndarray], out_path, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = reference_vs_blind_data(images, seed)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(data["radii"], data["reference_mae"], marker="o", label="reference-based")
    ax.plot(data["radii"], data["blind_mae"], marker="s", label="blind")
    ax.set_xlabel("true defocus radius (px)")
    ax.set_ylabel("mean absolute error (px)")
    ax.set_title("Reference PSD vs. blind defocus estimation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    mean_ref = float(np.mean(data["reference_mae"]))
    mean_blind = float(np.mean(data["blind_mae"]))
    gain = (mean_blind - mean_ref) / mean_blind if mean_blind > 1e-8 else float("nan")
    data["mean_reference_mae"] = mean_ref
    data["mean_blind_mae"] = mean_blind
    data["relative_error_reduction"] = gain
    return data


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--n-per-category", type=int, default=10)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.smoke and not dataset_available():
        print("MVTec AD not found. Re-run with --smoke, or set DIVIDE_DATA_ROOT.",
              file=sys.stderr)
        return 2

    if args.smoke:
        args.n_per_category = min(args.n_per_category, 4)
        args.size = min(args.size, 96)

    images = _pool_images(args.categories, args.n_per_category, args.size,
                          args.smoke, args.seed)
    print(f"loaded {len(images)} images from {args.categories}")

    figs = figures_dir()

    print("defect blindness...")
    d1 = make_defect_blindness_figure(images, figs / "dbde_defect_blindness.png", args.seed)
    print({k: [round(v, 4) for v in vs] for k, vs in d1.items()})

    print("parameter accuracy...")
    d2 = make_parameter_accuracy_figure(images, figs / "dbde_parameter_accuracy.png", args.seed)

    print("reference vs. blind...")
    d3 = make_reference_vs_blind_figure(images, figs / "dbde_reference_vs_blind.png", args.seed)
    print(f"  mean MAE: reference={d3['mean_reference_mae']:.3f}px  "
          f"blind={d3['mean_blind_mae']:.3f}px  "
          f"relative error reduction={d3['relative_error_reduction']:.1%}")

    print(f"\nfigures written to {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
