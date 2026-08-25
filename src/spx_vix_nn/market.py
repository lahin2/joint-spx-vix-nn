"""Synthetic jointly-consistent SPX / VIX book from a path-dependent vol truth model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .bs import implied_vol
from .config import Grid
from .models.pdv import PDVParams, simulate_pdv
from .vix import vix_strip


@dataclass
class VanillaQuote:
    expiry_days: int
    tau: float
    strike: float
    log_moneyness: float
    iv: float
    bid_ask: float
    call: bool
    price: float


@dataclass
class VixFutureQuote:
    obs_days: int
    tau: float
    level: float
    bid_ask: float


@dataclass
class MarketBook:
    spot: float
    rate: float
    spx: list[VanillaQuote]
    vix_futures: list[VixFutureQuote]
    vix_options: list[VanillaQuote]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot,
            "rate": self.rate,
            "spx": [asdict(q) for q in self.spx],
            "vix_futures": [asdict(q) for q in self.vix_futures],
            "vix_options": [asdict(q) for q in self.vix_options],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketBook":
        return cls(
            spot=payload["spot"],
            rate=payload["rate"],
            spx=[VanillaQuote(**q) for q in payload["spx"]],
            vix_futures=[VixFutureQuote(**q) for q in payload["vix_futures"]],
            vix_options=[VanillaQuote(**q) for q in payload["vix_options"]],
            meta=payload.get("meta", {}),
        )


def _otm_call_flag(log_m: float) -> bool:
    return log_m >= 0.0


def build_synthetic_book(
    grid: Grid | None = None,
    params: PDVParams | None = None,
    n_paths: int = 80_000,
    seed: int = 11,
    bid_ask_vol: float = 0.004,
) -> MarketBook:
    grid = grid or Grid()
    params = params or PDVParams()
    spots, inst_var = simulate_pdv(grid, params, n_paths=n_paths, seed=seed)

    spx_quotes: list[VanillaQuote] = []
    for day in grid.spx_expiry_days:
        tau = day * grid.dt
        s_t = spots[:, day]
        for k in grid.log_moneyness:
            strike = grid.spot * np.exp(k)
            call = _otm_call_flag(k)
            payoff = np.maximum(s_t - strike, 0.0) if call else np.maximum(strike - s_t, 0.0)
            px = float(np.exp(-grid.rate * tau) * payoff.mean())
            iv = float(implied_vol(px, grid.spot, strike, tau, grid.rate, call))
            if not np.isfinite(iv):
                continue
            spx_quotes.append(
                VanillaQuote(
                    expiry_days=day,
                    tau=tau,
                    strike=float(strike),
                    log_moneyness=float(k),
                    iv=iv,
                    bid_ask=bid_ask_vol * (1.0 + 2.0 * abs(k) / 0.18),
                    call=call,
                    price=px,
                )
            )

    vix_paths = vix_strip(inst_var, grid)
    fut_quotes: list[VixFutureQuote] = []
    vix_opt_quotes: list[VanillaQuote] = []
    for day in grid.vix_obs_days:
        tau = day * grid.dt
        vp = vix_paths[day]
        fut = float(vp.mean())
        fut_quotes.append(VixFutureQuote(obs_days=day, tau=tau, level=fut, bid_ask=0.003))
        if day != 21:
            continue
        for m in grid.vix_moneyness:
            strike = fut * m
            call = m >= 1.0
            payoff = np.maximum(vp - strike, 0.0) if call else np.maximum(strike - vp, 0.0)
            px = float(np.exp(-grid.rate * tau) * payoff.mean())
            iv = float(implied_vol(px, fut, strike, tau, grid.rate, call))
            if not np.isfinite(iv):
                continue
            vix_opt_quotes.append(
                VanillaQuote(
                    expiry_days=day,
                    tau=tau,
                    strike=float(strike),
                    log_moneyness=float(np.log(m)),
                    iv=iv,
                    bid_ask=0.008 * (1.0 + abs(m - 1.0) * 4.0),
                    call=call,
                    price=px,
                )
            )

    return MarketBook(
        spot=grid.spot,
        rate=grid.rate,
        spx=spx_quotes,
        vix_futures=fut_quotes,
        vix_options=vix_opt_quotes,
        meta={
            "truth": "pdv",
            "pdv": params.__dict__,
            "n_paths": n_paths,
            "seed": seed,
            "vix_definition": "pathwise_rms_instantaneous_variance",
        },
    )
