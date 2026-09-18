"""Environment-aware paths and config loading.

Never hardcode a data path anywhere else in this project. Import from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# environment detection
# --------------------------------------------------------------------------

def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").exists()


def on_colab() -> bool:
    return "COLAB_GPU" in os.environ


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root under which mvtec_ad/ and dtd/ live."""
    env = os.environ.get("DIVIDE_DATA_ROOT")
    if env:
        return Path(env)
    if on_kaggle():
        return Path("/kaggle/input")
    return repo_root() / "data"


def mvtec_root() -> Path:
    root = data_root()
    # Kaggle slugs directories; accept several spellings.
    for name in ("mvtec_ad", "mvtec-ad", "mvtecad", "MVTec_AD", "mvtec-anomaly-detection"):
        p = root / name
        if p.exists():
            # Kaggle sometimes nests one level deeper.
            if (p / "bottle").exists():
                return p
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "bottle").exists():
                    return child
            return p
    return root / "mvtec_ad"


def dtd_root() -> Path:
    root = data_root()
    for name in ("dtd", "DTD", "describable-textures", "dtd-dataset"):
        p = root / name
        if p.exists():
            if (p / "images").exists():
                return p
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "images").exists():
                    return child
            return p
    return root / "dtd"


def out_root() -> Path:
    """Where results and figures are written."""
    env = os.environ.get("DIVIDE_OUT_ROOT")
    if env:
        p = Path(env)
    elif on_kaggle():
        p = Path("/kaggle/working/divide_out")
    else:
        p = repo_root()
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = out_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    p = out_root() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoints_dir() -> Path:
    p = out_root() / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        cand = repo_root() / path
        if cand.exists():
            path = cand
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            f"Available configs: {sorted(p.name for p in (repo_root() / 'configs').glob('*.yaml'))}"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def require_data(kind: str) -> Path:
    """Fail loudly with an actionable message if a dataset is missing."""
    if kind == "mvtec":
        p = mvtec_root()
        ok = p.exists() and any((p / c).exists() for c in ("bottle", "carpet", "screw"))
        url = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
        hint = (
            "Accept the CC BY-NC-SA licence, download the tar, extract so that "
            f"{p}/bottle/train/good/*.png exists. On Kaggle, upload as a PRIVATE dataset."
        )
    elif kind == "dtd":
        p = dtd_root()
        ok = p.exists() and (p / "images").exists()
        url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
        hint = f"Extract so that {p}/images/<category>/*.jpg exists."
    else:
        raise ValueError(f"unknown dataset kind: {kind}")

    if not ok:
        raise FileNotFoundError(
            f"\n{'=' * 70}\nMissing dataset: {kind}\n"
            f"Expected at: {p}\nDownload:    {url}\n{hint}\n"
            f"Or set DIVIDE_DATA_ROOT to the directory containing it.\n{'=' * 70}"
        )
    return p


@dataclass
class RunPaths:
    """Bundle of output paths for one experiment run."""
    name: str
    results: Path = field(init=False)
    figures: Path = field(init=False)

    def __post_init__(self) -> None:
        self.results = results_dir()
        self.figures = figures_dir()

    def csv(self, stem: str) -> Path:
        return self.results / f"{stem}.csv"

    def fig(self, stem: str) -> Path:
        return self.figures / f"{stem}.png"
