"""Shared market grid and model constants.

Times are in year-fractions with 252 trading days. The VIX window is 21 days
(~ one month), matching the continuous quadratic-variation proxy used here
instead of the CBOE discrete log-contract strip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
VIX_WINDOW_DAYS = 21
VIX_DELTA = VIX_WINDOW_DAYS / TRADING_DAYS

# SPX option expiries (trading days) and VIX future/option dates.
SPX_EXPIRY_DAYS = (5, 21, 42)
VIX_OBS_DAYS = (5, 21, 42)

# Extra days so every VIX observation has a 21-day continuation.
HORIZON_DAYS = max(VIX_OBS_DAYS) + VIX_WINDOW_DAYS

LOG_MONEYNESS = (
    -0.18,
    -0.12,
    -0.08,
    -0.04,
    0.0,
    0.03,
    0.06,
    0.09,
)
VIX_MONEYNESS = (0.80, 0.90, 1.00, 1.10, 1.20)

SPOT = 1.0
RATE = 0.0


@dataclass
class Grid:
    dt: float = DT
    horizon_days: int = HORIZON_DAYS
    spx_expiry_days: tuple[int, ...] = SPX_EXPIRY_DAYS
    vix_obs_days: tuple[int, ...] = VIX_OBS_DAYS
    vix_window_days: int = VIX_WINDOW_DAYS
    log_moneyness: tuple[float, ...] = LOG_MONEYNESS
    vix_moneyness: tuple[float, ...] = VIX_MONEYNESS
    spot: float = SPOT
    rate: float = RATE

    @property
    def n_steps(self) -> int:
        return self.horizon_days

    @property
    def times(self) -> list[float]:
        return [i * self.dt for i in range(self.n_steps + 1)]


@dataclass
class TrainConfig:
    n_paths: int = 4096
    antithetic: bool = True
    seed: int = 7
    lr: float = 8e-3
    steps: int = 80
    grad_clip: float = 5.0
    w_spx: float = 1.0
    w_fut: float = 4.0
    w_vix: float = 1.5
    device: str = "cpu"
    hidden: int = 32
    extras: dict = field(default_factory=dict)
