"""Black-Scholes prices and implied-volatility inversion."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

_EPS = 1e-12


def _d1_d2(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, vol: np.ndarray, rate: float):
    vol = np.maximum(vol, _EPS)
    tau = np.maximum(tau, _EPS)
    fwd = spot * np.exp(rate * tau)
    m = np.log(np.maximum(fwd, _EPS) / np.maximum(strike, _EPS))
    sig_sqrt = vol * np.sqrt(tau)
    d1 = m / sig_sqrt + 0.5 * sig_sqrt
    d2 = d1 - sig_sqrt
    return d1, d2, fwd


def black_scholes_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    tau: float | np.ndarray,
    vol: float | np.ndarray,
    rate: float = 0.0,
    call: bool = True,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    vol = np.asarray(vol, dtype=float)
    d1, d2, fwd = _d1_d2(spot, strike, tau, vol, rate)
    df = np.exp(-rate * tau)
    call_px = df * (fwd * ndtr(d1) - strike * ndtr(d2))
    put_px = df * (strike * ndtr(-d2) - fwd * ndtr(-d1))
    return np.where(call, call_px, put_px)


def black_scholes_vega(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    tau: float | np.ndarray,
    vol: float | np.ndarray,
    rate: float = 0.0,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    vol = np.asarray(vol, dtype=float)
    d1, _, fwd = _d1_d2(spot, strike, tau, vol, rate)
    df = np.exp(-rate * tau)
    return df * fwd * np.sqrt(np.maximum(tau, _EPS)) * norm.pdf(d1)


def implied_vol(
    price: float | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    tau: float | np.ndarray,
    rate: float = 0.0,
    call: bool | np.ndarray = True,
    *,
    max_iter: int = 40,
    tol: float = 1e-8,
) -> np.ndarray:
    """Newton implied vol with a Brenner–Subrahmanyam seed and bracket fallback."""
    price = np.asarray(price, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    call_flag = np.asarray(call, dtype=bool)
    shape = np.broadcast_shapes(price.shape, np.shape(spot), strike.shape, tau.shape, call_flag.shape)
    price = np.broadcast_to(price, shape).copy()
    spot = np.broadcast_to(np.asarray(spot, dtype=float), shape)
    strike = np.broadcast_to(strike, shape)
    tau = np.broadcast_to(tau, shape)
    call_flag = np.broadcast_to(call_flag, shape)

    df = np.exp(-rate * tau)
    fwd = spot * np.exp(rate * tau)
    intrinsic = np.where(call_flag, np.maximum(df * (fwd - strike), 0.0), np.maximum(df * (strike - fwd), 0.0))
    upper = df * np.where(call_flag, fwd, strike)
    valid = (price > intrinsic + 1e-12) & (price < upper - 1e-12) & (tau > 1e-8)

    # ATM-ish seed; Brenner–Subrahmanyam for near-ATM, else 20%.
    seed = np.sqrt(2.0 * np.pi / np.maximum(tau, _EPS)) * np.maximum(price, _EPS) / np.maximum(spot, _EPS)
    vol = np.clip(np.where(valid, seed, np.nan), 0.01, 3.0)

    for _ in range(max_iter):
        px = black_scholes_price(spot, strike, tau, np.nan_to_num(vol, nan=0.2), rate, call_flag)
        vega = black_scholes_vega(spot, strike, tau, np.nan_to_num(vol, nan=0.2), rate)
        diff = px - price
        step = diff / np.maximum(vega, 1e-8)
        vol = np.clip(vol - step, 1e-4, 5.0)
        if np.all((~valid) | (np.abs(diff) < tol)):
            break

    vol = np.where(valid, vol, np.nan)
    return vol
