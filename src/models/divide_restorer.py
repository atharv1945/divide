"""DIVIDE itself as a Restorer - DBDE estimate -> PCIM -> (optional SARG blend).

Needs a trained PCIM checkpoint (and optionally a trained SARG checkpoint).
Neither has ever been trained to convergence - training is explicitly GPU
work (see README's cost table). src.experiments.train_pcim now exists and
writes checkpoints/pcim.pt; this module just hasn't had a real one to load
yet. This module exists so the "divide" restorer name is registered and
behaves like every other unavailable deep method: it fails loudly naming
the exact missing file, rather than being silently absent from
get_restorer()/available_restorers().

Uses DBDE's blind (no reference) blur detection, not the reference-based
primary path - this Restorer's interface is just img -> img with no
category context to look up a cached reference_psd() against. Training
(train_pcim.py) uses the reference path, since it does have category
context; if this restorer needs to match that accuracy for real inference,
it will need a category argument threaded through, which the eval grid /
demo callers don't currently pass.
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


def _restore(img: np.ndarray) -> np.ndarray:
    if not divide_available():
        raise RuntimeError(_missing_message())

    import torch
    from src.dbde.estimator import estimate
    from src.models.pcim import PCIM

    est = estimate(img)
    model = PCIM()
    state = torch.load(checkpoints_dir() / PCIM_WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    y = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    illum = torch.from_numpy(est.illum_field).unsqueeze(0).unsqueeze(0).float()
    # None when no blur was detected - PCIM.forward() skips the Wiener data
    # step entirely for None rather than deconvolving against a fake
    # identity kernel; see pcim.py's forward() docstring.
    kernel = (torch.from_numpy(est.kernel()).unsqueeze(0).unsqueeze(0).float()
             if est.blur_kind != "none" else None)
    sigma = torch.tensor([max(est.noise_sigma, 1e-4)], dtype=torch.float32)

    sarg_path = checkpoints_dir() / SARG_WEIGHTS
    with torch.no_grad():
        x_full, x_cons = model(y, illum_field=illum, kernel=kernel, sigma=sigma)
        if sarg_path.exists():
            from src.models.sarg import SARG
            sarg = SARG()
            sarg.load_state_dict(torch.load(sarg_path, map_location="cpu"))
            sarg.eval()
            x_blend, _ = sarg(y, x_full, x_cons)
        else:
            x_blend = x_full

    out = x_blend.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    return out.astype(np.float32)


def build_divide_restorer():
    from src.models.restorers import Restorer
    return Restorer("divide", _restore, tier="deep", needs_gpu=True)
