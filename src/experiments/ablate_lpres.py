"""L_pres ablation: does the preservation loss actually do anything?

Trains PCIM twice - once with L_pres in the optimized total, once without -
from IDENTICAL seed, data order, initialization, and step count, differing
in exactly that one loss term. Reports relative DRR for both and the delta.

This delta is the single most important number in the project: it's the
direct evidence that L_pres, not just the structural x_cons path, is doing
something. An airtight comparison matters more here than almost anywhere
else in the codebase, so the "identical except for use_lpres" guarantee is
asserted in code (_assert_only_use_lpres_differs), not just intended by
convention - if someone edits one variant's config and not the other, this
raises instead of silently producing a comparison that isn't apples-to-
apples.

Reuses src.experiments.train_pcim.run() for both variants, which is already
resumable (see its own module docstring) - so this script is resumable for
free: re-running it after an interruption just resumes whichever of the two
runs was mid-flight, exactly as train_pcim.py itself would.

    python -m src.experiments.ablate_lpres --config configs/train_pcim_cpu.yaml --smoke
    python -m src.experiments.ablate_lpres --config configs/train_pcim_cpu.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.experiments.train_pcim import _apply_smoke_overrides, default_device, run
from src.utils.paths import load_config, results_dir


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _assert_only_use_lpres_differs(cfg_a: dict, cfg_b: dict) -> None:
    flat_a, flat_b = _flatten(cfg_a), _flatten(cfg_b)
    diff_keys = {k for k in flat_a if flat_a.get(k) != flat_b.get(k)}
    diff_keys |= {k for k in flat_b if flat_a.get(k) != flat_b.get(k)}
    allowed = {"loss.use_lpres"}
    extra = diff_keys - allowed
    if extra:
        raise AssertionError(
            f"ablation configs differ in more than loss.use_lpres: {sorted(extra)}. "
            f"The L_pres delta is only meaningful if this is the ONLY difference - "
            f"fix whichever config drifted before trusting the comparison."
        )
    if flat_a.get("loss.use_lpres") == flat_b.get("loss.use_lpres"):
        raise AssertionError("both variants have the same loss.use_lpres - not an ablation")


def build_variant_configs(base_cfg: dict) -> tuple[dict, dict]:
    on = copy.deepcopy(base_cfg)
    off = copy.deepcopy(base_cfg)
    on["loss"]["use_lpres"] = True
    off["loss"]["use_lpres"] = False
    _assert_only_use_lpres_differs(on, off)
    return on, off


def run_ablation(base_cfg: dict, run_prefix: str, device: str | None = None,
                 smoke: bool = False, quiet: bool = False) -> dict:
    on_cfg, off_cfg = build_variant_configs(base_cfg)

    on_name = f"{run_prefix}_lpres_on"
    off_name = f"{run_prefix}_lpres_off"

    on_final = run(on_cfg, on_name, device=device, resume=True, smoke=smoke, quiet=quiet)
    off_final = run(off_cfg, off_name, device=device, resume=True, smoke=smoke, quiet=quiet)

    result = dict(
        lpres_on=on_final, lpres_off=off_final,
        relative_drr_delta=on_final["relative_drr"] - off_final["relative_drr"],
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-prefix", default="ablate_lpres")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = _apply_smoke_overrides(cfg)
    device = args.device or default_device()

    print(f"device : {device}")
    print(f"smoke  : {args.smoke}")
    print("training WITH L_pres...")
    print("training WITHOUT L_pres...")

    result = run_ablation(cfg, args.run_prefix, device=device, smoke=args.smoke, quiet=True)

    out_path = results_dir() / f"{args.run_prefix}_result.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("L_pres ABLATION")
    print("=" * 60)
    print(f"relative DRR, L_pres ON  : {result['lpres_on']['relative_drr']:.4f}")
    print(f"relative DRR, L_pres OFF : {result['lpres_off']['relative_drr']:.4f}")
    print(f"delta (on - off)         : {result['relative_drr_delta']:+.4f}")
    print(f"\nresult : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
