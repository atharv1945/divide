import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.experiments.compare_lpres_bootstrap import (
    _mean_field, _ratio_of_means, bootstrap_delta, compare,
)
from src.utils.paths import load_config


def test_ratio_of_means_matches_direct_computation():
    rows = [dict(drr_method=0.5, drr_id=0.5), dict(drr_method=0.3, drr_id=0.6)]
    expected = (0.5 + 0.3) / (0.5 + 0.6)
    assert _ratio_of_means(rows) == pytest.approx(expected, rel=1e-9)


def test_ratio_of_means_ignores_non_finite_rows():
    rows = [dict(drr_method=0.5, drr_id=0.5), dict(drr_method=float("nan"), drr_id=0.6)]
    assert _ratio_of_means(rows) == pytest.approx(1.0, abs=1e-9)


def test_mean_field_basic():
    rows = [dict(x=1.0), dict(x=3.0)]
    assert _mean_field(rows, "x") == pytest.approx(2.0)


def test_bootstrap_delta_point_estimate_and_ci_shape():
    on_rows = [dict(x=1.0)] * 20
    off_rows = [dict(x=0.5)] * 20
    result = bootstrap_delta(on_rows, off_rows, lambda r: _mean_field(r, "x"), n_boot=200, seed=0)
    assert result["point"] == pytest.approx(0.5, abs=1e-9)
    assert result["ci_lo"] == pytest.approx(0.5, abs=1e-9)  # zero variance in this fixture
    assert result["ci_hi"] == pytest.approx(0.5, abs=1e-9)
    assert result["excludes_zero"] is True


def test_bootstrap_delta_straddles_zero_when_no_real_difference():
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 1, size=40)
    on_rows = [dict(x=float(v)) for v in shared]
    off_rows = [dict(x=float(v)) for v in shared]  # identical - true delta is exactly 0
    result = bootstrap_delta(on_rows, off_rows, lambda r: _mean_field(r, "x"), n_boot=500, seed=0)
    assert result["point"] == pytest.approx(0.0, abs=1e-9)
    assert result["excludes_zero"] is False


def test_bootstrap_delta_empty_or_mismatched_is_nan_not_a_crash():
    result = bootstrap_delta([], [], lambda r: _mean_field(r, "x"))
    assert np.isnan(result["point"])
    result2 = bootstrap_delta([dict(x=1.0)], [dict(x=1.0), dict(x=2.0)], lambda r: _mean_field(r, "x"))
    assert np.isnan(result2["point"])


def test_compare_runs_end_to_end_smoke(tmp_path, monkeypatch):
    import src.experiments.train_pcim as tp
    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    ckpt_dir.mkdir(); res_dir.mkdir()
    monkeypatch.setattr(tp, "checkpoints_dir", lambda: ckpt_dir)
    monkeypatch.setattr(tp, "results_dir", lambda: res_dir)

    cfg = load_config("configs/train_pcim_cpu.yaml")
    cfg = tp._apply_smoke_overrides(cfg)
    tp.run(cfg, "cmp_on", device="cpu", resume=False, smoke=True, quiet=True)
    tp.run(cfg, "cmp_off", device="cpu", resume=False, smoke=True, quiet=True)

    result = compare(cfg, "cmp_on", "cmp_off", "cpu", seed=1, n=6, n_boot=20, smoke=True)
    assert "overall" in result
    for kind_result in result.values():
        assert "relative_drr_delta" in kind_result
        assert "residual_corr_delta" in kind_result
        assert "psnr_delta" in kind_result
        assert "ci_lo" in kind_result["relative_drr_delta"]
