"""Frozen-detector harness - PatchCore, PaDiM, Reverse Distillation, EfficientAD.

Wraps anomalib behind one interface: fit ONCE on clean normals, then score
any number of images with no further training. This is the whole point of
DIVIDE - the detector never sees restored or degraded images during fitting,
so any AUROC change downstream is attributable to what happened to the image
before it reached the detector, not to the detector adapting to it.

Verified end-to-end on CPU with the synthetic smoke images in
src/data/mvtec.py. PaDiM (fit + score, both classes, correct map shape) is
reliable and is what test_harness.py runs on every CPU pass. PatchCore and
ReverseDistillation each fit and score correctly at least once in an
isolated process during development, but this exact sandbox (Python 3.14,
which PyTorch itself warns is unsupported - see the `torch.jit.script`
FutureWarning at import) produced intermittent, non-reproducible failures
across repeated runs of the identical code - almost certainly environment
instability, not a DIVIDE bug, but never fully root-caused. Two concrete,
reproducible constraints found along the way:

  - image_size should be a multiple of 32. ReverseDistillation's
    encoder/decoder needs matching spatial sizes at every scale after
    repeated /2 downsampling; 48 fails with a tensor-size mismatch, 64
    works. PaDiM and PatchCore don't have this constraint.
  - EfficientAd additionally requires the Imagenette dataset (its loss
    regularises the student against natural images) - it downloads this
    itself on first fit if missing, so it needs network access even though
    every other model here only needs the category's own normal images.

Getting anomalib 1.1.0 to import and run at all on a current environment
took FOUR separate compatibility shims (see _compat.py for exactly what's
patched and why - none of it is DIVIDE logic, it's dependency-freshness
archaeology forced by the two-year gap between anomalib's last release and
today's numpy/pandas/rich/matplotlib). Re-run the isolated single-model
probes on the actual GPU machine (ordinary Python 3.10-3.12) before trusting
PatchCore/ReverseDistillation/EfficientAd results - see HANDOFF.md.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detect._compat import apply as _apply_compat

AVAILABLE = ["patchcore", "padim", "reverse_distillation", "efficientad"]

# EfficientAd's teacher/student distillation step requires batch size 1 -
# anomalib raises ValueError otherwise. Every other model is fine batched.
_TRAIN_BATCH_SIZE = {"efficientad": 1}


def _build_model(name: str):
    _apply_compat()
    from anomalib.models import EfficientAd, Padim, Patchcore, ReverseDistillation

    table = {
        "patchcore": Patchcore,
        "padim": Padim,
        "reverse_distillation": ReverseDistillation,
        "efficientad": EfficientAd,
    }
    if name not in table:
        raise KeyError(f"unknown detector {name!r}. known: {AVAILABLE}")
    return table[name]()


def _write_png01(img: np.ndarray, path: Path) -> None:
    u8 = np.clip(np.asarray(img, np.float32) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


@dataclass
class DetectionResult:
    score: float                    # image-level anomaly score, higher = more anomalous
    anomaly_map: np.ndarray         # (H, W) float32, same convention


class DetectorHarness:
    """One frozen anomalib detector for one category.

    fit() is meant to be called exactly once, on clean training normals.
    score() can then be called any number of times on any image - degraded,
    restored, whatever - without the detector adapting to it.
    """

    def __init__(self, name: str, image_size: int = 256, seed: int = 0,
                 max_epochs: int = 1, work_root: str | Path | None = None):
        if name not in AVAILABLE:
            raise KeyError(f"unknown detector {name!r}. known: {AVAILABLE}")
        self.name = name
        self.image_size = image_size
        self.seed = seed
        self.max_epochs = max_epochs
        self._work_root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="divide_detect_"))
        self._work_root.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._engine = None
        self._fitted = False

    def fit(self, normal_images: list[np.ndarray]) -> None:
        if not normal_images:
            raise ValueError("fit() needs at least one normal image")
        if self._fitted:
            raise RuntimeError(
                f"{self.name} detector already fitted - the whole point of the "
                f"frozen-detector harness is fit-once. Build a new DetectorHarness "
                f"instead of re-fitting this one."
            )
        _apply_compat()
        from anomalib.data import Folder
        from anomalib.engine import Engine
        import anomalib.engine.engine as _eng_mod

        data_dir = self._work_root / "train_data"
        normal_dir = data_dir / "normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(normal_images):
            _write_png01(img, normal_dir / f"{i:05d}.png")

        # No real defective images at fit time by construction (this is the
        # "fit on clean normals only" guarantee). test_split_mode="synthetic"
        # would let anomalib calibrate a normalization/threshold using
        # pseudo-anomalies synthesised from the normals themselves - tempting,
        # since it never leaks real defect information - but its imgaug-based
        # augmentation pipeline triggered an intermittent
        # `TypeError: only 0-dimensional arrays can be converted to Python
        # scalars` from PIL.ImageOps deep in anomalib's synthetic-anomaly
        # generator (reproduced non-deterministically: identical code, same
        # input, sometimes passes, sometimes doesn't - see HANDOFF.md). Skip
        # calibration entirely - can't: Lightning still runs a validation
        # pass at the end of every fit epoch regardless (it's how PatchCore/
        # PaDiM's on_validation_start -> self.fit() memory-bank construction
        # actually gets triggered), and val_split_mode="none" leaves no
        # val_data for it to run on (AttributeError: 'Folder' object has no
        # attribute 'val_data'). "from_dir" with no abnormal_dir carves the
        # validation split out of normal_dir itself (normal_split_ratio,
        # default 0.2) instead of synthesising anomalies - no imgaug
        # involved, so the bug above doesn't trigger. The resulting
        # normal-only validation metrics are meaningless and unused; we only
        # need SOME val_data to exist so the memory-bank fit step runs.
        datamodule = Folder(
            name=f"{self.name}_{data_dir.name}",
            root=data_dir,
            normal_dir="normal",
            # "classification": no mask_dir at fit time (no real defects
            # seen by construction). anomaly_maps are still produced at
            # inference regardless - task only gates pixel-level
            # threshold/metric computation, which fit() doesn't need.
            # "segmentation" + test_split_mode="from_dir" raises even with
            # zero abnormal images, since it unconditionally requires a
            # mask_dir for that split mode.
            task="classification",
            image_size=(self.image_size, self.image_size),
            train_batch_size=_TRAIN_BATCH_SIZE.get(self.name, 8),
            eval_batch_size=8,
            num_workers=0,
            test_split_mode="from_dir",
            val_split_mode="same_as_test",
            seed=self.seed,
        )

        model = _build_model(self.name)
        engine_dir = self._work_root / "engine"
        # anomalib versions the run dir with a "latest" symlink, which needs
        # elevated privileges on Windows outside Developer Mode. Harmless on
        # Linux (the actual GPU target); skip the symlink everywhere so the
        # CPU smoke path works unmodified on both.
        _eng_mod.create_versioned_dir = lambda root_dir: Path(root_dir) / "v0"
        engine = Engine(max_epochs=self.max_epochs, accelerator="cpu", devices=1,
                        default_root_dir=str(engine_dir), logger=False)
        engine.fit(model=model, datamodule=datamodule)

        self._model = model
        self._engine = engine
        self._fitted = True

    def score(self, image: np.ndarray) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError(f"{self.name} detector has not been fit() yet")
        _apply_compat()
        from anomalib.data import Folder
        import anomalib.engine.engine as _eng_mod
        _eng_mod.create_versioned_dir = lambda root_dir: Path(root_dir) / "v0"

        # Deliberately NOT using anomalib.data.PredictDataset here: it takes
        # an `image_size` argument that is silently a no-op unless a
        # `transform` is also passed (see PredictDataset.__getitem__ - it
        # only ever applies self.transform, never self.image_size), and
        # passing the model's own post-fit `.transform` explicitly hits a
        # second bug (an Albumentations-vs-tensor calling-convention
        # mismatch) on at least PatchCore/ReverseDistillation/EfficientAd in
        # this anomalib/torchvision combination. Routing every query image
        # through a one-image Folder datamodule instead reuses the exact
        # datamodule construction already exercised at fit time - image_size
        # is threaded through correctly there for all four models.
        with tempfile.TemporaryDirectory(dir=self._work_root) as td:
            query_dir = Path(td) / "query"
            query_dir.mkdir()
            _write_png01(image, query_dir / "query.png")
            # normal_dir is a required arg but its contents are irrelevant
            # here - predict() reads the TEST split, which normal_test_dir
            # pins to exactly the one query image regardless of what's in
            # normal_dir.
            dm = Folder(
                name=f"{self.name}_query", root=Path(td),
                normal_dir="query", normal_test_dir="query",
                # "classification" - no mask directory available for an
                # arbitrary query image. anomaly_maps are still produced;
                # only pixel-level threshold/metric computation needs a mask.
                task="classification", image_size=(self.image_size, self.image_size),
                train_batch_size=1, eval_batch_size=1, num_workers=0,
                test_split_mode="from_dir", val_split_mode="same_as_test",
            )
            preds = self._engine.predict(model=self._model, datamodule=dm)

        batch = preds[0]
        score = float(batch["pred_scores"][0])
        amap = batch["anomaly_maps"][0]
        amap = amap.squeeze().detach().cpu().numpy().astype(np.float32)
        if amap.shape != (self.image_size, self.image_size):
            # PatchCore's map resolution follows the backbone's feature-
            # pyramid stride and does not always exactly equal image_size
            # (e.g. 48 -> 42 at some strides). Resize so every detector
            # returns maps callers can compare against the same anomaly mask.
            amap = cv2.resize(amap, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)
        return DetectionResult(score=score, anomaly_map=amap)

    def close(self) -> None:
        """Release the on-disk work directory. Safe to call more than once."""
        shutil.rmtree(self._work_root, ignore_errors=True)
