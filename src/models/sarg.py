"""SARG - Sparse Anomaly-Residual Guard.

Unrolled Robust PCA (LISTA-style): decomposes the residual between the
degraded input and PCIM's aggressive reconstruction into a low-rank part
(structured degradation the model is confident about) and a sparse part
(anything localised and hard to explain away - which is exactly what a
surface defect looks like to this kind of decomposition, along with actual
sensor noise outliers). The sparse component's support becomes a protection
mask: where it's active, blend toward x_cons (the structurally content-
agnostic PCIM output) instead of x_full.

SARG is deliberately over-inclusive - high recall, low precision. It doesn't
try to be a detector (that's the frozen anomalib models' job); it only has to
avoid FALSE NEGATIVES, because a missed defect gets the full learned
restoration and is exactly the failure mode DIVIDE exists to prevent. A false
positive just means a normal-looking patch gets the (still perfectly
reasonable) conservative reconstruction instead of the aggressive one - cheap
insurance, not a hard failure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _soft_threshold(x: torch.Tensor, lam: torch.Tensor | float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)


def _svt(x: torch.Tensor, lam: torch.Tensor | float) -> torch.Tensor:
    """Singular-value thresholding, applied per (batch, channel) matrix."""
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    s = torch.clamp(s - lam, min=0.0)
    return u @ torch.diag_embed(s) @ vh


class SARG(nn.Module):
    """Unrolled RPCA producing a soft protection mask in [0, 1].

    Iterates:
        L_{t+1} = SVT(residual - S_t, lam_low)      # low-rank update
        S_{t+1} = soft_threshold(residual - L_{t+1}, lam_sparse)   # sparse update

    lam_low and lam_sparse are learned scalars (LISTA-style: a handful of
    parameters controlling threshold strength, not a full network) so the
    split can be calibrated during training, but the iteration STRUCTURE is
    fixed RPCA - this is not a free-form segmentation network.
    """

    def __init__(self, n_iters: int = 4, init_lam_low: float = 0.1,
                init_lam_sparse: float = 0.1, mask_gain: float = 4.0,
                mask_bias: float = -0.5):
        super().__init__()
        if not (3 <= n_iters <= 5):
            raise ValueError(f"n_iters should be 3-5 per spec, got {n_iters}")
        self.n_iters = n_iters
        self.log_lam_low = nn.Parameter(torch.tensor(float(init_lam_low)).log())
        self.log_lam_sparse = nn.Parameter(torch.tensor(float(init_lam_sparse)).log())
        # sigmoid(mask_gain * |S| + mask_bias): pushes the mask toward being
        # over-inclusive (high recall) rather than sharply thresholded.
        self.mask_gain = mask_gain
        self.mask_bias = mask_bias

    def decompose(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """residual: (B,C,H,W). Returns (low_rank, sparse), same shape.

        SVD operates on the (H,W) matrix per (batch, channel) slice.
        """
        b, c, h, w = residual.shape
        r = residual.reshape(b * c, h, w)
        lam_low = self.log_lam_low.exp()
        lam_sparse = self.log_lam_sparse.exp()

        L = torch.zeros_like(r)
        S = torch.zeros_like(r)
        for _ in range(self.n_iters):
            L = _svt(r - S, lam_low)
            S = _soft_threshold(r - L, lam_sparse)

        return L.reshape(b, c, h, w), S.reshape(b, c, h, w)

    def mask(self, residual: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) residual -> (B,1,H,W) soft protection mask in [0,1].

        Max over channels: a defect visible in any one channel should
        protect that pixel, not get averaged away by quiet channels.
        """
        _, sparse = self.decompose(residual)
        energy = sparse.abs().amax(dim=1, keepdim=True)
        return torch.sigmoid(self.mask_gain * energy + self.mask_bias)

    def forward(self, y: torch.Tensor, x_full: torch.Tensor,
               x_cons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend x_full and x_cons using the protection mask derived from
        (y - x_full) - where the aggressive reconstruction diverges most
        from the observation is where it most plausibly erased something.

        Returns (x_blend, mask).
        """
        residual = y - x_full
        m = self.mask(residual)
        x_blend = (1.0 - m) * x_full + m * x_cons
        return x_blend, m
