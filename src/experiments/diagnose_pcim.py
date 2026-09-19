"""Diagnostic: is a bad PCIM result the learned prox's fault, or the physics
chain's fault?

Evaluates x_cons (alpha=0, learned prox gated off - illumination divide,
Anscombe VST, closed-form Wiener HQS, inverse VST, nothing learned in the
loop) on the SAME eval set training used, plus the raw degraded input as a
baseline, and reports DRemR/PSNR for each alongside x_full for reference.

If x_cons's DRemR is also strongly negative, the physics chain itself is
broken - the learned prox isn't the problem, and reweighting L_pres won't
fix it. If x_cons's DRemR is near zero or positive, the physics is sound and
whatever is wrong is downstream of the learned prox / L_pres.

    python -m src.experiments.diagnose_pcim --config configs/train_pcim_cpu.yaml --run-name train_pcim_cpu
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.experiments.train_pcim import (
    _restore_both, build_eval_set, checkpoint_path, load_data_pools,
)
from src.degrade.anomaly import TextureBank
from src.metrics.core import degradation_removal_ratio, psnr
from src.models.pcim import PCIM
from src.utils.paths import dtd_root, load_config


@torch.no_grad()
def diagnose(cfg: dict, run_name: str, device: str = "cpu", smoke: bool = False) -> dict:
    ckpt = torch.load(checkpoint_path(run_name), map_location=device)
    model_cfg = cfg["model"]
    model = PCIM(n_iters=model_cfg["n_iters"], rho0=model_cfg["rho0"],
                rho_scale=model_cfg["rho_scale"], prox_width=model_cfg["prox_width"],
                prox_depth=model_cfg["prox_depth"], illum_clamp=model_cfg["illum_clamp"],
                vst_gain=model_cfg["vst_gain"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pools = load_data_pools(cfg, smoke=smoke)
    bank = TextureBank(dtd_root() if dtd_root().exists() else None)
    eval_set = build_eval_set(cfg, pools, bank)

    dremr_full, psnr_full = [], []
    dremr_cons, psnr_cons = [], []
    dremr_deg, psnr_deg = [], []

    for ex in eval_set:
        x_full_a, x_cons_a, _, _, _, _, y_a, _ = _restore_both(model, ex, device)
        r_full = x_full_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        r_cons = x_cons_a.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        normal_mask = ~ex.mask.astype(bool)

        d = degradation_removal_ratio(r_full, ex.y_a, ex.x_a, ex.mask)
        if np.isfinite(d):
            dremr_full.append(d)
        p = psnr(r_full, ex.x_a, mask=normal_mask)
        if np.isfinite(p):
            psnr_full.append(p)

        d = degradation_removal_ratio(r_cons, ex.y_a, ex.x_a, ex.mask)
        if np.isfinite(d):
            dremr_cons.append(d)
        p = psnr(r_cons, ex.x_a, mask=normal_mask)
        if np.isfinite(p):
            psnr_cons.append(p)

        d = degradation_removal_ratio(ex.y_a, ex.y_a, ex.x_a, ex.mask)  # trivially 0, kept for symmetry
        if np.isfinite(d):
            dremr_deg.append(d)
        p = psnr(ex.y_a, ex.x_a, mask=normal_mask)
        if np.isfinite(p):
            psnr_deg.append(p)

    def agg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return dict(
        n=len(eval_set),
        x_full_dremr=agg(dremr_full), x_full_psnr=agg(psnr_full),
        x_cons_dremr=agg(dremr_cons), x_cons_psnr=agg(psnr_cons),
        degraded_dremr=agg(dremr_deg), degraded_psnr=agg(psnr_deg),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = diagnose(cfg, args.run_name, device=args.device)

    print(f"n eval examples : {out['n']}")
    print()
    print(f"{'':12s} {'DRemR':>10s} {'PSNR (dB)':>10s}")
    print(f"{'degraded':12s} {out['degraded_dremr']:>10.3f} {out['degraded_psnr']:>10.3f}")
    print(f"{'x_cons':12s} {out['x_cons_dremr']:>10.3f} {out['x_cons_psnr']:>10.3f}")
    print(f"{'x_full':12s} {out['x_full_dremr']:>10.3f} {out['x_full_psnr']:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
