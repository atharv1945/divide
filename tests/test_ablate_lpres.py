import copy

import pytest

torch = pytest.importorskip("torch")

from src.experiments.ablate_lpres import (
    _assert_only_use_lpres_differs, build_variant_configs, run_ablation,
)
from src.experiments.train_pcim import _apply_smoke_overrides
from src.utils.paths import load_config


@pytest.fixture
def cfg():
    c = load_config("configs/train_pcim_cpu.yaml")
    return _apply_smoke_overrides(c)


def test_build_variant_configs_differ_only_in_use_lpres(cfg):
    on, off = build_variant_configs(cfg)
    assert on["loss"]["use_lpres"] is True
    assert off["loss"]["use_lpres"] is False
    on2 = copy.deepcopy(on)
    on2["loss"]["use_lpres"] = off["loss"]["use_lpres"]
    assert on2 == off  # every other key identical


def test_assert_raises_when_configs_drift_beyond_use_lpres(cfg):
    on, off = build_variant_configs(cfg)
    off["train"]["lr"] = on["train"]["lr"] * 10  # simulate accidental drift
    with pytest.raises(AssertionError, match="more than loss.use_lpres"):
        _assert_only_use_lpres_differs(on, off)


def test_assert_raises_when_use_lpres_is_the_same_in_both(cfg):
    on = copy.deepcopy(cfg)
    off = copy.deepcopy(cfg)
    on["loss"]["use_lpres"] = True
    off["loss"]["use_lpres"] = True
    with pytest.raises(AssertionError, match="not an ablation"):
        _assert_only_use_lpres_differs(on, off)


def test_assert_passes_for_a_true_use_lpres_only_diff(cfg):
    on = copy.deepcopy(cfg)
    off = copy.deepcopy(cfg)
    on["loss"]["use_lpres"] = True
    off["loss"]["use_lpres"] = False
    _assert_only_use_lpres_differs(on, off)  # must not raise


def test_run_ablation_smoke_end_to_end(cfg, tmp_path, monkeypatch):
    import src.experiments.train_pcim as tp
    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    ckpt_dir.mkdir(); res_dir.mkdir()
    monkeypatch.setattr(tp, "checkpoints_dir", lambda: ckpt_dir)
    monkeypatch.setattr(tp, "results_dir", lambda: res_dir)
    import src.experiments.ablate_lpres as ab
    monkeypatch.setattr(ab, "results_dir", lambda: res_dir)

    result = run_ablation(cfg, "abtest", device="cpu", smoke=True, quiet=True)

    assert "relative_drr" in result["lpres_on"]
    assert "relative_drr" in result["lpres_off"]
    assert "relative_drr_delta" in result

    # both variants actually got separate checkpoints
    assert (ckpt_dir / "abtest_lpres_on.pt").exists()
    assert (ckpt_dir / "abtest_lpres_off.pt").exists()

    # resuming (identical call) should not error and should not restart either run
    result2 = run_ablation(cfg, "abtest", device="cpu", smoke=True, quiet=True)
    assert result2["lpres_on"]["n"] == result["lpres_on"]["n"]
