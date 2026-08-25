"""Quadratic-variation VIX proxy.

CBOE VIX is a discrete log-contract strip. Here VIX on a path at observation
date t is the RMS of instantaneous variance over the next 21 trading days:

    VIX_t = sqrt( (1/Δ) ∫_t^{t+Δ} σ_u² du )   (pathwise Riemann sum)

That is a realized-vol proxy, not the true (conditional-expectation) VIX.
The synthetic book and the neural SDE use the same definition, so the joint
loss is internally consistent. See the README for the convexity gap versus
E[sqrt(E[QV | F_t])].
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Grid


def vix_pathwise_np(inst_var: np.ndarray, obs_idx: int, window: int) -> np.ndarray:
    """inst_var: [n_paths, n_steps] variance over each step following time i."""
    slice_ = inst_var[:, obs_idx : obs_idx + window]
    return np.sqrt(np.maximum(slice_.mean(axis=1), 1e-16))


def vix_pathwise_torch(inst_var: torch.Tensor, obs_idx: int, window: int) -> torch.Tensor:
    slice_ = inst_var[:, obs_idx : obs_idx + window]
    return torch.sqrt(torch.clamp(slice_.mean(dim=1), min=1e-16))


def vix_strip(inst_var: np.ndarray, grid: Grid) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for day in grid.vix_obs_days:
        out[day] = vix_pathwise_np(inst_var, day, grid.vix_window_days)
    return out
