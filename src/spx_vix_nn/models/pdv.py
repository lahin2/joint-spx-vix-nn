"""Path-dependent volatility truth model used to mint the synthetic book.

The variance factor is an EWMA of Brownian shocks (a discrete PDV / rough-Bergomi
cousin). Leverage on the spot creates a steep short-dated SPX put skew without
the CIR vol-of-vol that inflates VIX convexity in Heston.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Grid


@dataclass
class PDVParams:
    sigma0: float = 0.18
    kappa: float = 8.0
    vol_of_vol: float = 1.15
    rho: float = -0.72
    leverage: float = -1.4
    v_floor: float = 1e-6


def simulate_pdv(
    grid: Grid,
    params: PDVParams,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spots [n, n_steps+1] and step instantaneous variance [n, n_steps]."""
    rng = np.random.default_rng(seed)
    n_steps = grid.n_steps
    dt = grid.dt
    sqrt_dt = np.sqrt(dt)

    z1 = rng.standard_normal((n_paths, n_steps))
    z_ind = rng.standard_normal((n_paths, n_steps))
    z2 = params.rho * z1 + np.sqrt(1.0 - params.rho**2) * z_ind

    log_s = np.full(n_paths, np.log(grid.spot))
    x = np.zeros(n_paths)
    spots = np.empty((n_paths, n_steps + 1))
    inst_var = np.empty((n_paths, n_steps))
    spots[:, 0] = grid.spot
    decay = np.exp(-params.kappa * dt)

    for i in range(n_steps):
        # Path-dependent factor: EWMA of vol shocks, plus a mild spot leverage term.
        log_m = log_s - np.log(grid.spot)
        sigma = params.sigma0 * np.exp(
            params.vol_of_vol * x + params.leverage * np.clip(log_m, -0.4, 0.4)
        )
        sigma = np.maximum(sigma, np.sqrt(params.v_floor))
        inst_var[:, i] = sigma**2
        log_s = log_s + (grid.rate - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z1[:, i]
        x = decay * x + np.sqrt((1.0 - decay**2) / max(2.0 * params.kappa, 1e-8)) * z2[:, i]
        # Stationary OU increment variance is (1-e^{-2κΔ})/(2κ); we use a unit-vol OU.
        spots[:, i + 1] = np.exp(log_s)

    return spots, inst_var
