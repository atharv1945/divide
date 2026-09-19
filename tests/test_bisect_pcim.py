import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.mvtec import synthetic_split
from src.utils.paths import load_config


@pytest.fixture
def smoke_cfg():
    import src.experiments.train_pcim as tp
    cfg = load_config("configs/train_pcim_cpu.yaml")
    return tp._apply_smoke_overrides(cfg)


def _one_image(size=64):
    train, _ = synthetic_split("carpet", n_train=1, n_test_good=1, n_test_bad=1, size=size, seed=0)
    return train[0].image


def test_bisect_runs_and_stage0_matches_identity_baseline(smoke_cfg):
    from src.experiments.bisect_pcim import bisect
    out = bisect(smoke_cfg, device="cpu")
    assert set(out) == {"stage0", "stage1", "stage2", "stage3"}
    for name, r in out.items():
        assert "dremr" in r and "psnr" in r and "n" in r
    assert out["stage0"]["dremr"] == pytest.approx(0.0, abs=1e-4)


def test_vst_roundtrip_check_passes_well_under_tolerance():
    from src.experiments.bisect_pcim import check_vst_roundtrip
    img = _one_image()
    for row in check_vst_roundtrip(img):
        assert row["passes_tol"]
        assert row["max_abs_err"] < 1e-4


def test_wiener_known_kernel_improves_on_no_deconv_at_reasonable_nsr():
    from src.experiments.bisect_pcim import check_wiener_known_kernel
    img = _one_image(size=96)
    out = check_wiener_known_kernel(img, true_radius=3.0)
    best_known = max(row["psnr"] for row in out["known_kernel"])
    assert best_known > out["blurred_vs_clean_psnr"]


def test_wiener_phase_check_reports_small_subpixel_offset():
    from src.experiments.bisect_pcim import check_wiener_phase
    img = _one_image(size=96)
    out = check_wiener_phase(img, true_radius=3.0, nsr=0.01)
    assert abs(out["offset_y_px"]) < 2.0
    assert abs(out["offset_x_px"]) < 2.0


def test_illumination_stats_reports_expected_keys():
    from src.experiments.bisect_pcim import check_illumination_stats
    imgs = [_one_image(size=64) for _ in range(3)]
    out = check_illumination_stats(imgs, clamp=4.0)
    for k in ("field_min", "field_max", "field_mean", "frac_clamped"):
        assert k in out
    assert 0.0 <= out["frac_clamped"] <= 1.0


def test_inspect_eval_set_estimates_runs(smoke_cfg):
    from src.experiments.bisect_pcim import inspect_eval_set_estimates
    out = inspect_eval_set_estimates(smoke_cfg)
    assert out["n"] > 0
    assert "blur_kind_counts" in out
    assert np.isfinite(out["sigma_mean"])


def test_wiener_actual_pipeline_sigma_runs():
    from src.experiments.bisect_pcim import check_wiener_actual_pipeline_sigma
    img = _one_image(size=96)
    out = check_wiener_actual_pipeline_sigma(img, true_radius=3.0, actual_sigma=0.01)
    assert set(out) == {"rho=1.0", "rho=2.0", "rho=4.0", "rho=8.0", "rho=16.0"}
    for row in out.values():
        assert "effective_reg" in row and "psnr" in row
