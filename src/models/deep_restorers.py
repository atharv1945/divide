"""Deep restorer loaders - NAFNet, Restormer, DiffBIR.

None of these can run on this machine: no GPU, no cloned repos, no weights.
Every code path here is exercised on CPU only through its FAILURE mode - given
missing weights, `build_deep_restorer` must raise RuntimeError with the exact
download URL and target path, never silently skip the restorer. That failure
path is unit-tested. The construction and inference code past that point
follows each upstream repo's own documented CLI as closely as I could verify
from its README (see HANDOFF.md for sources), but has never executed. Treat
it as a well-informed first draft, not a working integration, until someone
with a GPU runs it once and reports back.

Design choice: rather than re-importing each repo's internal model classes
(NAFNet and Restormer are BasicSR-based, with APIs that have moved between
versions; DiffBIR is a full latent-diffusion pipeline with samplers and a
ControlNet), each restorer is invoked through the CLI script the upstream repo
already tests and documents - `demo.py` / `inference.py` - via subprocess,
round-tripping a single image through a temp file. This is slower than an
in-process call but far more likely to actually work on a repo whose internals
I cannot execute to verify.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.utils.paths import checkpoints_dir, third_party_dir

REPO_SPECS = {
    "nafnet": dict(
        git="https://github.com/megvii-research/NAFNet.git",
        weights="NAFNet-SIDD-width64.pth",
        weights_url="https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view",
        weights_rel_path="experiments/pretrained_models/NAFNet-SIDD-width64.pth",
        note=("Google Drive - cannot be fetched programmatically, download by hand. "
              "The cloned repo also needs its own BasicSR fork installed once: "
              "`cd third_party/nafnet && python setup.py develop --no_cuda_ext`."),
    ),
    "restormer": dict(
        git="https://github.com/swz30/Restormer.git",
        weights="real_denoising.pth",
        weights_url="https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view",
        weights_rel_path="Denoising/pretrained_models/real_denoising.pth",
        note="Google Drive - download by hand into the path above.",
    ),
    "diffbir": dict(
        git="https://github.com/XPixelGroup/DiffBIR.git",
        weights="v2.pth",
        weights_url="https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
        weights_rel_path="weights/v2.pth",
        base_weights="v2-1_512-ema-pruned.ckpt",
        base_weights_url=("https://huggingface.co/stabilityai/stable-diffusion-2-1-base/"
                          "resolve/main/v2-1_512-ema-pruned.ckpt"),
        base_weights_rel_path="weights/v2-1_512-ema-pruned.ckpt",
        note=("Hugging Face - CAN be fetched with wget/curl, unlike NAFNet/Restormer's "
              "Google Drive links. Needs BOTH v2.pth AND the SD v2.1 base checkpoint. "
              "Slow: full diffusion sampling per image (~50 steps), expect seconds per "
              "image on a GPU, not the sub-second cost of NAFNet/Restormer."),
    ),
}


def _weights_missing_message(name: str) -> str:
    spec = REPO_SPECS[name]
    lines = [
        f"\n{'=' * 70}",
        f"Restorer {name!r} has no weights available.",
        f"Download: {spec['weights_url']}",
        f"Save as:  {checkpoints_dir() / spec['weights']}",
        f"{spec['note']}",
    ]
    if "base_weights" in spec:
        lines += [
            f"Also needs: {spec['base_weights_url']}",
            f"Save as:    {checkpoints_dir() / spec['base_weights']}",
        ]
    lines += [
        "Re-run once both files are in place.",
        f"To run without {name!r}, drop it from the config's `restorers` list.",
        "=" * 70,
    ]
    return "\n".join(lines)


def _check_weights(name: str) -> None:
    spec = REPO_SPECS[name]
    missing = not (checkpoints_dir() / spec["weights"]).exists()
    if "base_weights" in spec:
        missing = missing or not (checkpoints_dir() / spec["base_weights"]).exists()
    if missing:
        raise RuntimeError(_weights_missing_message(name))


def _repo_dir(name: str) -> Path:
    return third_party_dir() / name


def _ensure_repo(name: str) -> Path:
    """Clone the upstream code repo if it isn't already vendored.

    Unlike weights, these are ordinary public git repos - fetchable
    programmatically, so this is not a "fail loud and stop" path. A clone
    failure (no network, repo moved) still raises, with the command that
    failed, but that's a different and much rarer failure mode than the
    weights case above.
    """
    spec = REPO_SPECS[name]
    dest = _repo_dir(name)
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", spec["git"], str(dest)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not clone {spec['git']} into {dest}.\n"
            f"stderr:\n{e.stderr}\n"
            f"Clone it by hand and re-run, or check network access."
        ) from e
    return dest


def _stage_weights(name: str) -> None:
    """Copy weights from checkpoints/ into the path the repo's own CLI expects."""
    spec = REPO_SPECS[name]
    repo = _repo_dir(name)

    def _copy(fname: str, rel: str) -> None:
        src = checkpoints_dir() / fname
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    _copy(spec["weights"], spec["weights_rel_path"])
    if "base_weights" in spec:
        _copy(spec["base_weights"], spec["base_weights_rel_path"])


def _read_png01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"restorer produced no output at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _write_png01(img: np.ndarray, path: Path) -> None:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def _run_cli(cmd: list[str], cwd: Path, name: str) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{name} inference failed.\ncmd: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        ) from e


def _restore_nafnet(img: np.ndarray) -> np.ndarray:
    repo = _repo_dir("nafnet")
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.png"
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        _write_png01(img, in_path)
        _run_cli([
            sys.executable, "basicsr/demo.py",
            "-opt", "options/test/SIDD/NAFNet-width64.yml",
            "--input_path", str(in_path),
            "--output_path", str(out_dir / "out.png"),
        ], cwd=repo, name="nafnet")
        return _read_png01(out_dir / "out.png")


def _restore_restormer(img: np.ndarray) -> np.ndarray:
    repo = _repo_dir("restormer")
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir(); out_dir.mkdir()
        _write_png01(img, in_dir / "in.png")
        _run_cli([
            sys.executable, "demo.py",
            "--task", "Real_Denoising",
            "--input_dir", str(in_dir),
            "--result_dir", str(out_dir),
        ], cwd=repo, name="restormer")
        outs = list(out_dir.glob("*.png"))
        if not outs:
            raise RuntimeError(f"restormer produced no output in {out_dir}")
        return _read_png01(outs[0])


def _restore_diffbir(img: np.ndarray) -> np.ndarray:
    repo = _repo_dir("diffbir")
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir(); out_dir.mkdir()
        _write_png01(img, in_dir / "in.png")
        # "sr" with upscale 1 is DiffBIR's generic blind-restoration mode -
        # the closest fit to the mixed blur/illumination/noise/jpeg
        # degradations DIVIDE evaluates, none of which are super-resolution.
        _run_cli([
            sys.executable, "inference.py",
            "--task", "sr", "--upscale", "1", "--version", "v2",
            "--sampler", "spaced", "--steps", "50", "--captioner", "none",
            "--pos_prompt", "", "--neg_prompt",
            "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
            "--cfg_scale", "4",
            "--input", str(in_dir), "--output", str(out_dir),
            "--device", "cuda", "--precision", "fp32",
        ], cwd=repo, name="diffbir")
        outs = list(out_dir.rglob("*.png"))
        if not outs:
            raise RuntimeError(f"diffbir produced no output in {out_dir}")
        return _read_png01(outs[0])


_RESTORE_FN = {
    "nafnet": _restore_nafnet,
    "restormer": _restore_restormer,
    "diffbir": _restore_diffbir,
}


def build_deep_restorer(name: str):
    """Return a Restorer for `name`, or raise RuntimeError naming the exact fix.

    Weights are checked FIRST, before touching the network, so a missing-
    weights report never depends on being online.
    """
    from src.models.restorers import Restorer  # local import, avoids a cycle

    if name not in REPO_SPECS:
        raise KeyError(f"unknown deep restorer {name!r}. known: {sorted(REPO_SPECS)}")

    _check_weights(name)
    _ensure_repo(name)
    _stage_weights(name)

    fn = _RESTORE_FN[name]
    return Restorer(name, fn, tier="deep", needs_gpu=True)
