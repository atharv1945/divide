import numpy as np
import pytest

from src.models.restorers import (
    CLASSICAL, DEEP_SPECS, available_restorers, get_restorer,
)
from src.models.deep_restorers import REPO_SPECS, _weights_missing_message


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
def test_deep_restorer_without_weights_fails_loudly_with_url(name):
    """No weights are present in this environment - this IS the code path
    that must run on the GPU machine's first invocation too, before weights
    are downloaded. It must never silently no-op."""
    r = get_restorer(name)
    assert r.tier == "deep"
    with pytest.raises(RuntimeError) as exc:
        r(_img())
    msg = str(exc.value)
    assert REPO_SPECS[name]["weights_url"] in msg
    assert REPO_SPECS[name]["weights"] in msg


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
    assert "no training script has been written" in msg.lower()


def test_divide_is_listed_in_available_restorers():
    assert "divide" in available_restorers(include_deep=True)
    assert "divide" not in available_restorers(include_deep=False)
