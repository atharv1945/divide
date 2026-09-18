"""DIVIDE itself as a Restorer - DBDE estimate -> PCIM -> (optional SARG blend).

Needs a trained PCIM checkpoint (and optionally a trained SARG checkpoint).
Neither has ever been trained - PCIM training is explicitly GPU work (see
README's cost table) and no training script has been written yet either
(losses.py/pcim.py/sarg.py are the building blocks; wiring a training loop
around degrade_pair() counterfactual pairs is the next piece of glue, left
for the GPU machine - see HANDOFF.md). This module exists so the "divide"
restorer name is registered and behaves like every other unavailable deep
method: it fails loudly naming the exact missing file, rather than being
silently absent from get_restorer()/available_restorers().
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
        f"project's own model. Train it with src.models.pcim.PCIM and the "
        f"losses in src.models.losses (L_rec/L_deg/L_freq/L_pres), using "
        f"counterfactual pairs from src.degrade.simulator.degrade_pair(). "
        f"No training script has been written yet; see HANDOFF.md for what "
        f"that needs to wire together.\n"
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
    kernel = torch.from_numpy(est.kernel()).unsqueeze(0).unsqueeze(0).float()
    sigma = torch.tensor([est.noise_sigma], dtype=torch.float32)

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
