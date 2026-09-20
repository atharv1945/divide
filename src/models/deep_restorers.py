"""Deep restorer loaders - NAFNet, Restormer, DiffBIR.

NAFNet and Restormer run in-process on CPU: the architecture is loaded and
its weights read from disk ONCE (via functools.lru_cache, keyed on nothing -
there is only one config per restorer here), then reused for every image the
go/no-go study or eval grid throws at it. This replaced an earlier subprocess-
per-image design (each call shelled out to the upstream repo's demo.py CLI,
reloading the ~450MB/~100MB checkpoint from scratch every time) once
"load once, reuse across cells" became a hard requirement - the CLI design
was never going to hit that on CPU.

NAFNet's architecture is vendored (src/models/_nafnet_arch.py) rather than
imported from the cloned repo: NAFNet_arch.py's `from basicsr.utils import
get_root_logger` pulls in basicsr's full utils package (lmdb/tensorboard/
wandb-touching) for one logger function this module never calls. See that
file's docstring for the license/provenance note. Restormer's architecture
has no such transitive dependency (only `einops`), so it's loaded directly
from the cloned repo via `runpy.run_path` on the single self-contained arch
file - no package-level `import basicsr` needed, no risk of colliding with
NAFNet's differently-vendored `basicsr` package if both were ever imported
in the same process.

DiffBIR is EXCLUDED from this study for compute reasons: it's a full
latent-diffusion pipeline (~50 sampling steps per image) with no realistic
CPU path at this scope - the note in REPO_SPECS below estimates seconds per
image on a GPU, and this machine has none. Its loader is left in place
(REPO_SPECS entry, weights-check, subprocess CLI path) in case a GPU becomes
available later; `build_deep_restorer("diffbir")` still fails loud with the
weights URL rather than silently doing something else.
"""
from __future__ import annotations

import functools
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils.paths import checkpoints_dir, third_party_dir

REPO_SPECS = {
    "nafnet": dict(
        git="https://github.com/megvii-research/NAFNet.git",
        weights="NAFNet-SIDD-width64.pth",
        weights_url="https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view",
        weights_rel_path="experiments/pretrained_models/NAFNet-SIDD-width64.pth",
        note=("Google Drive - not fetchable with a plain GET (redirect + confirm-token "
              "page for files this size). `pip install gdown` then "
              "`gdown 'https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR' "
              f"-O {{checkpoints_dir}}/NAFNet-SIDD-width64.pth` handles the token; a plain "
              "browser download works too."),
    ),
    "restormer": dict(
        git="https://github.com/swz30/Restormer.git",
        weights="real_denoising.pth",
        weights_url="https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view",
        weights_rel_path="Denoising/pretrained_models/real_denoising.pth",
        note="Google Drive - see nafnet's note above, same fetch method (gdown or by hand).",
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
              "EXCLUDED from the CPU go/no-go study on compute grounds, not a technical "
              "blocker: full diffusion sampling (~50 steps) per image, expect seconds per "
              "image on a GPU - this machine has none, and the projected CPU cost is far "
              "past what a multi-day run can absorb. Re-enable by adding 'diffbir' back to "
              "a study's --restorers list once a GPU is available."),
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

    Only restormer's build path needs this (its architecture is loaded via
    runpy.run_path straight from the cloned repo's arch file - see module
    docstring). NAFNet's architecture is vendored, so its build path never
    calls this. Unlike weights, a repo is an ordinary public git clone - not
    a "fail loud and stop" path; a clone failure still raises, with the
    command that failed, but that's rarer than the weights case above.
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


# --------------------------------------------------------------------------
# NAFNet / Restormer - in-process, loaded once, CPU
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _load_nafnet() -> torch.nn.Module:
    _check_weights("nafnet")
    from src.models._nafnet_arch import NAFNet
    # Params match options/test/SIDD/NAFNet-width64.yml in the upstream
    # repo - the exact config the downloaded checkpoint was trained under.
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(checkpoints_dir() / REPO_SPECS["nafnet"]["weights"], map_location="cpu")
    model.load_state_dict(ckpt["params"], strict=True)
    model.eval()
    return model


@functools.lru_cache(maxsize=None)
def _load_restormer() -> torch.nn.Module:
    _check_weights("restormer")
    _ensure_repo("restormer")
    arch_path = _repo_dir("restormer") / "basicsr" / "models" / "archs" / "restormer_arch.py"
    mod = runpy.run_path(str(arch_path))
    # Params match demo.py's Real_Denoising branch - the task the
    # downloaded real_denoising.pth checkpoint was trained for.
    params = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                  num_refinement_blocks=4, heads=[1, 2, 4, 8],
                  ffn_expansion_factor=2.66, bias=False,
                  LayerNorm_type="BiasFree", dual_pixel_task=False)
    model = mod["Restormer"](**params)
    ckpt = torch.load(checkpoints_dir() / REPO_SPECS["restormer"]["weights"], map_location="cpu")
    model.load_state_dict(ckpt["params"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _restore_nafnet(img: np.ndarray) -> np.ndarray:
    """NAFNet.forward() pads to its own padder_size and crops back
    internally (see check_image_size in the vendored arch) - no manual
    padding needed here."""
    model = _load_nafnet()
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()
    y = model(x)
    return y.squeeze(0).permute(1, 2, 0).numpy()


@torch.no_grad()
def _restore_restormer(img: np.ndarray) -> np.ndarray:
    """Restormer's forward does NOT self-pad (unlike NAFNet) - replicates
    demo.py's manual pad-to-multiple-of-8-then-crop."""
    model = _load_restormer()
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()
    h, w = x.shape[-2:]
    m = 8
    H = ((h + m) // m) * m if h % m else h
    W = ((w + m) // m) * m if w % m else w
    x = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode="reflect")
    y = model(x)
    y = torch.clamp(y, 0, 1)[:, :, :h, :w]
    return y.squeeze(0).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------
# DiffBIR - excluded from CPU study, subprocess CLI path kept for a GPU box
# --------------------------------------------------------------------------

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


def _stage_diffbir_weights() -> None:
    spec = REPO_SPECS["diffbir"]
    repo = _repo_dir("diffbir")

    def _copy(fname: str, rel: str) -> None:
        src = checkpoints_dir() / fname
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    _copy(spec["weights"], spec["weights_rel_path"])
    if "base_weights" in spec:
        _copy(spec["base_weights"], spec["base_weights_rel_path"])


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

    Weights are checked FIRST, before touching the network or a repo clone,
    so a missing-weights report never depends on being online. NAFNet and
    Restormer then load their model ONCE here (functools.lru_cache) - the
    returned Restorer's fn is a plain forward pass, not a reload.
    """
    from src.models.restorers import Restorer  # local import, avoids a cycle

    if name not in REPO_SPECS:
        raise KeyError(f"unknown deep restorer {name!r}. known: {sorted(REPO_SPECS)}")

    if name == "nafnet":
        _load_nafnet()  # raises with the download message if weights are absent
    elif name == "restormer":
        _load_restormer()
    else:
        _check_weights(name)
        _ensure_repo(name)
        _stage_diffbir_weights()

    fn = _RESTORE_FN[name]
    needs_gpu = name == "diffbir"
    return Restorer(name, fn, tier="deep", needs_gpu=needs_gpu)
