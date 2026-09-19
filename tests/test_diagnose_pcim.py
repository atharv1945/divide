import copy

import pytest

torch = pytest.importorskip("torch")

from src.utils.paths import load_config


def test_diagnose_runs_and_reports_all_three_baselines(tmp_path, monkeypatch):
    import src.experiments.train_pcim as tp
    import src.experiments.diagnose_pcim as dp

    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    ckpt_dir.mkdir(); res_dir.mkdir()
    monkeypatch.setattr(tp, "checkpoints_dir", lambda: ckpt_dir)
    monkeypatch.setattr(tp, "results_dir", lambda: res_dir)
    monkeypatch.setattr(dp, "checkpoint_path", tp.checkpoint_path)

    cfg = load_config("configs/train_pcim_cpu.yaml")
    cfg = tp._apply_smoke_overrides(cfg)
    tp.run(cfg, "diag_test", device="cpu", resume=False, smoke=True, quiet=True)

    out = dp.diagnose(cfg, "diag_test", device="cpu", smoke=True)

    assert out["n"] > 0
    for key in ("x_full_dremr", "x_full_psnr", "x_cons_dremr", "x_cons_psnr",
               "degraded_dremr", "degraded_psnr"):
        assert key in out

    # DRemR of the raw degraded input against itself is 0 by construction
    # (see degradation_removal_ratio's docstring: 0.0 = nothing done) -
    # this is the diagnostic's own baseline sanity check, not a claim about
    # the model.
    assert out["degraded_dremr"] == pytest.approx(0.0, abs=1e-6)
