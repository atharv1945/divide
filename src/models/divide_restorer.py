"""DIVIDE itself as a Restorer - DBDE estimate -> PCIM -> (optional SARG blend).

Two entry points:

  build_divide_restorer()             - the generic "divide" name, looks for
                                         checkpoints/pcim.pt. Registered so
                                         "divide" behaves like every other
                                         unavailable deep method (fails
                                         loudly naming the exact missing
                                         file) rather than being silently
                                         absent from get_restorer().
  build_divide_restorer_from_run(...) - loads a SPECIFIC named training run's
                                         checkpoint (e.g. the L_pres
                                         ablation's "ablate_lpres_lpres_on"/
                                         "_off") via the same cfg/run_name
                                         convention train_pcim.py and
                                         eval_pcim_holdout.py use, for
                                         studies that need more than one
                                         trained DIVIDE variant side by
                                         side.

Uses DBDE's blind (no reference) blur detection, not the reference-based
primary path - this Restorer's interface is just img -> img with no
category context to look up a cached reference_psd() against. Training
(train_pcim.py) uses the reference path, since it does have category
context; every other restorer in this registry that touches DBDE (wiener,
classical_pipeline) has this exact same limitation, so it's not a DIVIDE-
specific handicap in a grid that compares them - all deconvolving restorers
here run blind detection, not just this one.
"""
from __future__ import annotations

import numpy as np

from src.utils.paths import checkpoints_dir

PCIM_WEIGHTS = "pcim.pt"
SARG_WEIGHTS = "sarg.pt"  # optional


def _missing_message() -> str:
    return (
        f"\n{'=' * 70}\n"
        f"Restorer 'divide' has no trained PCIM checkpoint.\n"
        f"Expected at: {checkpoints_dir() / PCIM_WEIGHTS}\n"
        f"No pretrained weights exist to download for this one - it's this "
        f"project's own model. Train it with:\n"
        f"  python -m src.experiments.train_pcim --config configs/train_pcim_gpu.yaml\n"
        f"To run without it, drop 'divide' from the config's `restorers` list.\n"
        f"{'=' * 70}"
    )


def divide_available() -> bool:
    return (checkpoints_dir() / PCIM_WEIGHTS).exists()


def _restore_with_model(model, img: np.ndarray, sarg_path=None) -> np.ndarray:
    """Shared inference path for any already-loaded PCIM model - both
    build_divide_restorer() and build_divide_restorer_from_run() call this,
    so there's exactly one place that builds the DBDE estimate, tensors,
    and does the optional SARG blend."""
    import torch
    from src.dbde.estimator import estimate

    est = estimate(img)
    y = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    illum = torch.from_numpy(est.illum_field).unsqueeze(0).unsqueeze(0).float()
    # None when no blur was detected - PCIM.forward() skips the Wiener data
    # step entirely for None rather than deconvolving against a fake
    # identity kernel; see pcim.py's forward() docstring.
    kernel = (torch.from_numpy(est.kernel()).unsqueeze(0).unsqueeze(0).float()
             if est.blur_kind != "none" else None)
    sigma = torch.tensor([max(est.noise_sigma, 1e-4)], dtype=torch.float32)

    with torch.no_grad():
        x_full, x_cons = model(y, illum_field=illum, kernel=kernel, sigma=sigma)
        if sarg_path is not None and sarg_path.exists():
            from src.models.sarg import SARG
            sarg = SARG()
            sarg.load_state_dict(torch.load(sarg_path, map_location="cpu"))
            sarg.eval()
            x_blend, _ = sarg(y, x_full, x_cons)
        else:
            x_blend = x_full

    out = x_blend.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    return out.astype(np.float32)


def _restore(img: np.ndarray) -> np.ndarray:
    if not divide_available():
        raise RuntimeError(_missing_message())

    import torch
    from src.models.pcim import PCIM

    model = PCIM()
    # train_pcim.py's save_checkpoint() wraps model/optimizer/step/rng/
    # scale_factors together - this used to load that whole dict AS the
    # model's state_dict directly (state = torch.load(...); model.
    # load_state_dict(state)), which would fail on any checkpoint that
    # script actually produces. Never caught because nothing had called
    # this against a real checkpoint until the grid needed to.
    ckpt = torch.load(checkpoints_dir() / PCIM_WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    sarg_path = checkpoints_dir() / SARG_WEIGHTS
    return _restore_with_model(model, img, sarg_path=sarg_path)


def build_divide_restorer():
    from src.models.restorers import Restorer
    return Restorer("divide", _restore, tier="deep", needs_gpu=True)


def build_divide_restorer_from_run(cfg: dict, run_name: str, restorer_name: str):
    """Loads a SPECIFIC training run's checkpoint (by run_name, same
    convention as train_pcim.py/eval_pcim_holdout.py) rather than the
    generic checkpoints/pcim.pt - e.g. the L_pres ablation's two variants,
    which need to exist as two independently-selectable restorers in the
    same study. Reuses eval_pcim_holdout.load_model() rather than
    reimplementing PCIM construction/checkpoint-loading a third time.
    Fails loudly (FileNotFoundError, not silently) if that run's
    checkpoint doesn't exist - torch.load's own error is already exact
    about the missing path, no need to wrap it further here.
    """
    from src.experiments.eval_pcim_holdout import load_model
    from src.models.restorers import Restorer

    model = load_model(cfg, run_name, "cpu")

    def _fn(img: np.ndarray) -> np.ndarray:
        return _restore_with_model(model, img, sarg_path=None)

    return Restorer(restorer_name, _fn, tier="deep", needs_gpu=False)
