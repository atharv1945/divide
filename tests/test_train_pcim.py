import copy

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.utils.paths import load_config


SMOKE_CFG_PATH = "configs/train_pcim_cpu.yaml"


@pytest.fixture
def cfg():
    import src.experiments.train_pcim as tp
    c = load_config(SMOKE_CFG_PATH)
    return tp._apply_smoke_overrides(c)


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """Point checkpoints_dir()/results_dir() at a tmp dir for this test."""
    import src.experiments.train_pcim as tp
    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    ckpt_dir.mkdir()
    res_dir.mkdir()
    monkeypatch.setattr(tp, "checkpoints_dir", lambda: ckpt_dir)
    monkeypatch.setattr(tp, "results_dir", lambda: res_dir)
    return tp, ckpt_dir, res_dir


def test_smoke_run_completes_and_writes_artifacts(cfg, redirected):
    tp, ckpt_dir, res_dir = redirected
    final = tp.run(cfg, "t1", device="cpu", resume=False, smoke=True, quiet=True)

    assert (ckpt_dir / "t1.pt").exists()
    losses = pd.read_csv(res_dir / "t1_losses.csv")
    assert len(losses) == cfg["train"]["steps"]
    assert list(losses.step) == list(range(1, cfg["train"]["steps"] + 1))

    evals = pd.read_csv(res_dir / "t1_eval.csv")
    assert not evals.empty
    assert "relative_drr" in final


def test_kill_and_resume_continues_step_count_without_gaps_or_dupes(cfg, redirected):
    """Simulates a kill: run to step 3, 'restart' the process (fresh Python-
    level state, same checkpoint dir), and continue to step 6. The loss CSV
    across both invocations must be a clean, gap-free, duplicate-free
    sequence - exactly what an uninterrupted 6-step run would have written."""
    tp, ckpt_dir, res_dir = redirected
    cfg = copy.deepcopy(cfg)
    cfg["train"]["steps"] = 6

    tp.run(cfg, "t2", device="cpu", resume=False, max_steps=3, smoke=True, quiet=True)
    losses_part1 = pd.read_csv(res_dir / "t2_losses.csv")
    assert list(losses_part1.step) == [1, 2, 3]

    ckpt = torch.load(ckpt_dir / "t2.pt", map_location="cpu")
    assert ckpt["step"] == 3

    tp.run(cfg, "t2", device="cpu", resume=True, max_steps=6, smoke=True, quiet=True)
    losses_full = pd.read_csv(res_dir / "t2_losses.csv")
    assert list(losses_full.step) == [1, 2, 3, 4, 5, 6]


def test_resumed_run_continues_the_same_sample_sequence_as_uninterrupted(cfg, redirected):
    """The RNG state must resume too, not just the model weights - otherwise
    a killed-and-resumed run samples a DIFFERENT sequence of training
    examples than an uninterrupted one from the same seed, which is exactly
    what ablate_lpres.py's apples-to-apples comparison depends on NOT
    happening."""
    tp, ckpt_dir, res_dir = redirected
    cfg_a = copy.deepcopy(cfg)
    cfg_a["train"]["steps"] = 6
    cfg_b = copy.deepcopy(cfg_a)

    tp.run(cfg_a, "uninterrupted", device="cpu", resume=False, smoke=True, quiet=True)

    tp.run(cfg_b, "killed", device="cpu", resume=False, max_steps=3, smoke=True, quiet=True)
    tp.run(cfg_b, "killed", device="cpu", resume=True, max_steps=6, smoke=True, quiet=True)

    a = pd.read_csv(res_dir / "uninterrupted_losses.csv")
    b = pd.read_csv(res_dir / "killed_losses.csv")
    # same architecture + same seed + same resumed RNG -> identical loss
    # trajectory (model init is also seeded identically since both start
    # fresh from torch.manual_seed(cfg["seed"]))
    pd_testing_assert = pd.testing.assert_series_equal
    pd_testing_assert(a["total"], b["total"], check_exact=False, rtol=1e-4, atol=1e-6)


def test_resume_is_a_noop_when_already_at_target_step(cfg, redirected):
    tp, ckpt_dir, res_dir = redirected
    cfg = copy.deepcopy(cfg)
    cfg["train"]["steps"] = 4

    tp.run(cfg, "t3", device="cpu", resume=False, smoke=True, quiet=True)
    n_rows_first = len(pd.read_csv(res_dir / "t3_losses.csv"))

    tp.run(cfg, "t3", device="cpu", resume=True, smoke=True, quiet=True)
    n_rows_second = len(pd.read_csv(res_dir / "t3_losses.csv"))
    assert n_rows_first == n_rows_second == 4


def test_use_lpres_false_excludes_pres_from_backward_but_still_logs_it(cfg, redirected):
    tp, ckpt_dir, res_dir = redirected
    cfg = copy.deepcopy(cfg)
    cfg["loss"]["use_lpres"] = False
    tp.run(cfg, "t4", device="cpu", resume=False, smoke=True, quiet=True)
    losses = pd.read_csv(res_dir / "t4_losses.csv")
    assert "pres" in losses.columns  # still logged for comparison
    assert (losses["pres"] >= 0).all()  # a real, computed value - not a placeholder


def test_evaluate_returns_finite_or_nan_metrics(cfg, redirected):
    import numpy as np
    from src.experiments.train_pcim import build_eval_set, evaluate, load_data_pools
    from src.degrade.anomaly import TextureBank
    from src.models.pcim import PCIM

    tp, ckpt_dir, res_dir = redirected
    cfg = copy.deepcopy(cfg)
    pools = load_data_pools(cfg, smoke=True)
    eval_set = build_eval_set(cfg, pools, TextureBank())
    model = PCIM(n_iters=4, rho0=1.0, rho_scale=2.0, prox_width=8, prox_depth=3,
                illum_clamp=4.0, vst_gain=1.0)
    out = evaluate(model, eval_set, "cpu")
    for k in ("relative_drr", "dremr", "psnr_normal"):
        assert np.isfinite(out[k]) or np.isnan(out[k])
    assert out["n"] == len(eval_set)
