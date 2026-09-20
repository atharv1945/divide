import pytest

torch = pytest.importorskip("torch")

from src.utils.paths import load_config


def _setup(tmp_path, monkeypatch):
    import src.experiments.train_pcim as tp
    import src.experiments.eval_pcim_holdout as eh

    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    ckpt_dir.mkdir(); res_dir.mkdir()
    monkeypatch.setattr(tp, "checkpoints_dir", lambda: ckpt_dir)
    monkeypatch.setattr(tp, "results_dir", lambda: res_dir)

    cfg = load_config("configs/train_pcim_cpu.yaml")
    cfg = tp._apply_smoke_overrides(cfg)
    tp.run(cfg, "holdout_test", device="cpu", resume=False, smoke=True, quiet=True)
    return cfg, tp, eh


def test_holdout_eval_and_ratio_check_run_end_to_end(tmp_path, monkeypatch):
    cfg, tp, eh = _setup(tmp_path, monkeypatch)

    model = eh.load_model(cfg, "holdout_test", "cpu")
    pools = tp.load_data_pools(cfg, smoke=True)
    from src.degrade.anomaly import TextureBank
    bank = TextureBank(None)
    ref_cache = tp.build_reference_cache(pools)

    eval_set = eh.build_holdout_eval_set(cfg, pools, bank, ref_cache, seed=1, n=6)
    assert len(eval_set) == 6
    rows = eh.per_example_eval(model, eval_set, "cpu")
    assert len(rows) == 6
    for key in ("category", "kind", "family", "severity", "den", "dremr", "psnr", "relative_drr"):
        assert key in rows[0]

    summ = eh.summarize(rows)
    assert summ["n"] == 6

    check = eh.ratio_instability_check(rows)
    assert "corr_den_dremr" in check and "corr_den_psnr" in check

    dist = eh.original_eval_den_distribution(cfg, pools, bank, ref_cache)
    assert dist["n"] == cfg["eval"]["n_examples"]
    assert dist["min"] <= dist["median"] <= dist["max"]


def test_degradation_magnitude_is_deterministic_given_fixed_inputs(tmp_path, monkeypatch):
    """original_eval_den_distribution() depends only on (degraded, clean)
    pairs from the fixed eval set, not on model weights - calling it twice
    with the same cfg/pools must give identical numbers regardless of which
    checkpoint (if any) is loaded."""
    cfg, tp, eh = _setup(tmp_path, monkeypatch)
    pools = tp.load_data_pools(cfg, smoke=True)
    from src.degrade.anomaly import TextureBank
    bank = TextureBank(None)
    ref_cache = tp.build_reference_cache(pools)

    d1 = eh.original_eval_den_distribution(cfg, pools, bank, ref_cache)
    d2 = eh.original_eval_den_distribution(cfg, pools, bank, ref_cache)
    assert d1 == d2
