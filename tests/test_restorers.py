import numpy as np
import pytest

from src.models.restorers import (
    CLASSICAL, DEEP_SPECS, available_restorers, get_restorer,
)
from src.models.deep_restorers import REPO_SPECS, _weights_missing_message
from src.utils.paths import checkpoints_dir


def _img(size=48):
    rng = np.random.default_rng(0)
    return np.clip(rng.random((size, size, 3)).astype(np.float32) * 0.6 + 0.2, 0, 1)


# ---------------------------------------------------------------- classical

@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_classical_restorer_preserves_shape_and_range(name):
    x = _img()
    r = get_restorer(name)
    out = r(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_identity_is_a_true_noop():
    x = _img()
    out = get_restorer("identity")(x)
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_unknown_restorer_raises_with_available_list():
    with pytest.raises(KeyError, match="unknown restorer"):
        get_restorer("not_a_real_restorer")


def test_available_restorers_lists_classical_and_deep():
    names = available_restorers(include_deep=True)
    assert set(CLASSICAL).issubset(names)
    assert set(DEEP_SPECS).issubset(names)


# ---------------------------------------------------------------- deep (fail-loud path)

@pytest.mark.parametrize("name", sorted(REPO_SPECS))
def test_deep_restorer_without_weights_fails_loudly_with_url(name, tmp_path, monkeypatch):
    """This IS the code path that must run on a fresh machine's first
    invocation, before weights are downloaded - it must never silently
    no-op. Monkeypatched to an empty tmp_path rather than relying on this
    dev machine actually lacking weights (nafnet/restormer's ARE present
    here now - see test_deep_restorer_with_real_weights_runs below).

    nafnet/restormer's loaders are functools.lru_cache'd with no arguments
    (load once, reuse forever within a process - see deep_restorers.py) so
    a real successful load anywhere earlier in this test SESSION (e.g.
    test_demo.py's restormer panel, which may well have already run) would
    make this test see a cached model and never re-check checkpoints_dir at
    all - cache_clear() first forces a genuine reload attempt under the
    monkeypatched (empty) directory."""
    import src.models.deep_restorers as dr
    monkeypatch.setattr(dr, "checkpoints_dir", lambda: tmp_path)
    dr._load_nafnet.cache_clear()
    dr._load_restormer.cache_clear()
    r = get_restorer(name)
    assert r.tier == "deep"
    with pytest.raises(RuntimeError) as exc:
        r(_img())
    msg = str(exc.value)
    assert REPO_SPECS[name]["weights_url"] in msg
    assert REPO_SPECS[name]["weights"] in msg
    # leave the cache clean so a later test doesn't see this tmp_path stuck
    # in place of the real checkpoints_dir (monkeypatch itself is undone
    # automatically at teardown, but the lru_cache is process-global state
    # monkeypatch doesn't know about).
    dr._load_nafnet.cache_clear()
    dr._load_restormer.cache_clear()


def _weights_present(name: str) -> bool:
    spec = REPO_SPECS[name]
    if not (checkpoints_dir() / spec["weights"]).exists():
        return False
    if "base_weights" in spec and not (checkpoints_dir() / spec["base_weights"]).exists():
        return False
    return True


@pytest.mark.parametrize("name", ["nafnet", "restormer"])
@pytest.mark.skipif(not any(_weights_present(n) for n in ("nafnet", "restormer")),
                    reason="neither nafnet nor restormer weights are downloaded")
def test_deep_restorer_with_real_weights_runs(name):
    """Once weights are present, these run real in-process CPU inference -
    no subprocess, no per-call reload (build_deep_restorer loads the model
    once; calling the returned Restorer twice must reuse the exact same
    cached module object, not rebuild it)."""
    if not _weights_present(name):
        pytest.skip(f"{name} weights not downloaded")
    from src.models.deep_restorers import _load_nafnet, _load_restormer
    loader = {"nafnet": _load_nafnet, "restormer": _load_restormer}[name]

    r = get_restorer(name)
    assert r.tier == "deep"
    x = _img(size=32)
    out = r(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0

    m1 = loader()
    m2 = loader()
    assert m1 is m2  # lru_cache - loaded once, reused


def test_deep_specs_urls_are_well_formed():
    for name, spec in REPO_SPECS.items():
        assert spec["git"].startswith("https://")
        assert spec["weights_url"].startswith("https://")
        assert spec["weights"]


def test_weights_missing_message_names_both_files_for_diffbir():
    msg = _weights_missing_message("diffbir")
    assert REPO_SPECS["diffbir"]["weights_url"] in msg
    assert REPO_SPECS["diffbir"]["base_weights_url"] in msg


def test_unknown_deep_restorer_name_raises_keyerror():
    from src.models.deep_restorers import build_deep_restorer
    with pytest.raises(KeyError):
        build_deep_restorer("not_a_real_model")


# ---------------------------------------------------------------- divide itself

def test_divide_restorer_without_checkpoint_fails_loudly():
    from src.models.divide_restorer import PCIM_WEIGHTS
    r = get_restorer("divide")
    assert r.tier == "deep"
    with pytest.raises(RuntimeError) as exc:
        r(_img())
    msg = str(exc.value)
    assert PCIM_WEIGHTS in msg
    assert "train_pcim" in msg.lower()


def test_divide_is_listed_in_available_restorers():
    assert "divide" in available_restorers(include_deep=True)
    assert "divide" not in available_restorers(include_deep=False)
