"""Compatibility shims needed to import anomalib at all.

anomalib 1.1.0 (last release before this project started, mid-2024) has two
dependency-freshness problems that have nothing to do with DIVIDE but will
block the frozen-detector harness on any modern environment:

1. `imgaug==0.4.0` (pulled in by anomalib's data augmentation pipeline)
   reads `np.sctypes` at import time, which NumPy removed in 2.0. Fails with
   `AttributeError: np.sctypes was removed in the NumPy 2.0 release` before
   a single line of our code runs. Fix: restore the dict imgaug expects.
   Prefer pinning numpy<2 in the environment if that's an option (it is not
   on every machine - see requirements.txt); this shim works regardless.

2. `anomalib.models.__init__` eagerly imports its (unrelated) video AI-VAD
   model, which uses a legacy `pkg_resources`-based CLIP loader. Fails with
   `ModuleNotFoundError: pkg_resources` on setuptools>=81, which dropped it.
   Fix: requirements.txt pins `setuptools<81`. Nothing to shim here - just
   documenting why that pin exists, since otherwise it looks arbitrary.

3. `anomalib`'s prediction-time visualization callback calls
   `FigureCanvasAgg.tostring_rgb()`, which recent Matplotlib (>=3.9-ish)
   dropped in favour of `buffer_rgba()`. Fails with
   `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'`
   the first time `Engine.predict()` runs (fit is unaffected - this callback
   only fires on predict). Fix: shim `tostring_rgb` in terms of
   `buffer_rgba()`, dropping the alpha channel.

4. PatchCore's coreset-subsampling progress bar
   (`anomalib.utils.rich.CacheRichLiveState`) reads the private attribute
   `Console._live`, which `rich` replaced with a `_live_stack` list at some
   point after anomalib 1.1.0 was released. Fails with
   `AttributeError: 'Console' object has no attribute '_live'` partway
   through PatchCore fitting (after the backbone forward passes - the
   coreset step is the last thing `fit()` does). Fix: patch
   `CacheRichLiveState.__enter__`/`__exit__` to read `_live_stack` instead.
   PaDiM, ReverseDistillation and EfficientAd never touch this code path -
   only PatchCore's coreset sampler calls `safe_track`.

Call `apply()` before importing anything from `anomalib`. Idempotent.
"""
from __future__ import annotations

import numpy as np


def apply() -> None:
    if not hasattr(np, "sctypes"):
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(FigureCanvasAgg, "tostring_rgb"):
        def _tostring_rgb(self):
            buf = np.asarray(self.buffer_rgba())
            return buf[..., :3].tobytes()
        FigureCanvasAgg.tostring_rgb = _tostring_rgb

    from rich.console import Console
    if not hasattr(Console, "_live"):
        from anomalib.utils.rich import CacheRichLiveState

        def _enter(self) -> None:
            with self.console._lock:
                stack = getattr(self.console, "_live_stack", None)
                self.live = stack[-1] if stack else None
                self.console.clear_live()

        def _exit(self, exc_type, exc_val, exc_tb) -> None:
            # __enter__ already popped `self.live` off _live_stack via
            # clear_live(); just push it back, don't pop again.
            if self.live:
                self.console.set_live(self.live)

        CacheRichLiveState.__enter__ = _enter
        CacheRichLiveState.__exit__ = _exit
