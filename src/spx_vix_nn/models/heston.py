"""Heston baseline: SPX-only calibration, then freeze and mark VIX.

Closed-form vanillas via the Gatheral/Albrecher characteristic function.
VIX uses the CIR affine map from instantaneous variance to 30-day quadratic
variation, sampled by simulating the variance CIR — so VIX is the true
conditional-expectation object, not the pathwise RMS proxy. That mismatch is
part of the puzzle exhibit (Heston is a different VIX ontology). For a fair
numeric comparison we also report Heston pathwise-QV VIX from the same Euler
scheme used by the neural SDE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

from ..bs import implied_vol
from ..config import Grid
from ..market import MarketBook
from ..vix import vix_strip


@dataclass
class HestonParams:
    v0: float = 0.035
    kappa: float = 1.4
    theta: float = 0.04
    xi: float = 0.85
    rho: float = -0.75
    rate: float = 0.0


def _cf(phi: complex, tau: float, params: HestonParams, j: int) -> complex:
    """Little-Heston-trap characteristic function piece f_j(phi)."""
    v0, kappa, theta, xi, rho, r = (
        params.v0,
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.rate,
    )
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    i = 1j
    d = np.sqrt((rho * xi * phi * i - b) ** 2 - xi**2 * (2 * u * phi * i - phi**2))
    g = (b - rho * xi * phi * i + d) / (b - rho * xi * phi * i - d)
    # Albrecher "little trap": use 1/g formulation for stability
    c = 1.0 / g
    exp_dt = np.exp(-d * tau)
    g_term = (1 - c * exp_dt) / (1 - c)
    C = r * phi * i * tau + (kappa * theta / xi**2) * ((b - rho * xi * phi * i - d) * tau - 2 * np.log(g_term))
    D = ((b - rho * xi * phi * i - d) / xi**2) * ((1 - exp_dt) / (1 - c * exp_dt))
    return np.exp(C + D * v0)


def heston_call(spot: float, strike: float, tau: float, params: HestonParams) -> float:
    if tau < 1e-8:
        return float(max(spot - strike, 0.0))

    log_k = np.log(strike)

    def integrand(phi: float, j: int) -> float:
        if phi == 0.0:
            return 0.0
        f = _cf(phi, tau, params, j)
        return np.real(np.exp(-1j * phi * log_k) * f / (1j * phi))

    def p(j: int) -> float:
        # Truncated Fourier integral; Heston integrand decays as 1/phi.
        val, _ = quad(lambda x: integrand(x, j), 1e-8, 200.0, limit=250, epsabs=1e-6)
        return 0.5 + val / np.pi

    df = np.exp(-params.rate * tau)
    fwd = spot * np.exp(params.rate * tau)
    p1 = p(1)
    p2 = p(2)
    return float(df * (fwd * p1 - strike * p2))


def heston_vanilla_iv(spot: float, strike: float, tau: float, params: HestonParams, call: bool) -> float:
    call_px = heston_call(spot, strike, tau, params)
    df = np.exp(-params.rate * tau)
    fwd = spot * np.exp(params.rate * tau)
    put_px = call_px - df * (fwd - strike)
    px = call_px if call else put_px
    iv = implied_vol(px, spot, strike, tau, params.rate, call)
    return float(iv) if np.isfinite(iv) else float("nan")


def simulate_heston_euler(
    grid: Grid,
    params: HestonParams,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_steps = grid.n_steps
    dt = grid.dt
    sqrt_dt = np.sqrt(dt)
    z1 = rng.standard_normal((n_paths, n_steps))
    z_ind = rng.standard_normal((n_paths, n_steps))
    z2 = params.rho * z1 + np.sqrt(max(1.0 - params.rho**2, 0.0)) * z_ind

    log_s = np.full(n_paths, np.log(grid.spot))
    v = np.full(n_paths, params.v0)
    spots = np.empty((n_paths, n_steps + 1))
    variance = np.empty((n_paths, n_steps + 1))
    inst_var = np.empty((n_paths, n_steps))
    spots[:, 0] = grid.spot
    variance[:, 0] = v
    for i in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        inst_var[:, i] = v_pos
        log_s = log_s + (grid.rate - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1[:, i]
        v = (
            v
            + params.kappa * (params.theta - v_pos) * dt
            + params.xi * np.sqrt(v_pos) * sqrt_dt * z2[:, i]
        )
        spots[:, i + 1] = np.exp(log_s)
        variance[:, i + 1] = np.maximum(v, 0.0)
    return spots, inst_var, variance


def cir_forward_variance(v_t: np.ndarray, params: HestonParams, delta: float) -> np.ndarray:
    if params.kappa < 1e-8:
        return v_t
    a = (1.0 - np.exp(-params.kappa * delta)) / (params.kappa * delta)
    b = params.theta * (1.0 - a)
    return a * v_t + b


def calibrate_heston_spx(
    book: MarketBook,
    grid: Grid | None = None,
    n_paths: int = 12_000,
    seed: int = 5,
) -> HestonParams:
    """Least-squares fit of Heston to SPX implied vols only (CRN Euler MC)."""
    grid = grid or Grid()
    quotes = [q for q in book.spx if np.isfinite(q.iv)]

    def pack(x: np.ndarray) -> HestonParams:
        return HestonParams(
            v0=float(np.clip(x[0] ** 2, 1e-4, 0.5)),
            kappa=float(np.clip(x[1] ** 2, 0.05, 12.0)),
            theta=float(np.clip(x[2] ** 2, 1e-4, 0.5)),
            xi=float(np.clip(x[3] ** 2, 0.05, 3.0)),
            rho=float(np.tanh(x[4])),
            rate=book.rate,
        )

    def objective(x: np.ndarray) -> float:
        p = pack(x)
        spots, _, _ = simulate_heston_euler(grid, p, n_paths=n_paths, seed=seed)
        err = 0.0
        for q in quotes:
            st = spots[:, q.expiry_days]
            payoff = np.maximum(st - q.strike, 0.0) if q.call else np.maximum(q.strike - st, 0.0)
            px = float(np.exp(-book.rate * q.tau) * payoff.mean())
            iv = implied_vol(px, book.spot, q.strike, q.tau, book.rate, q.call)
            if not np.isfinite(iv):
                return 1e3
            w = 1.0 / max(q.bid_ask, 1e-4)
            err += w * (iv - q.iv) ** 2
        return err / max(len(quotes), 1)

    x0 = np.array([np.sqrt(0.035), np.sqrt(1.8), np.sqrt(0.04), np.sqrt(0.95), np.arctanh(-0.75)])
    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 140, "xatol": 1e-3, "fatol": 1e-5})
    return pack(res.x)


def price_book_heston(
    book: MarketBook,
    params: HestonParams,
    grid: Grid | None = None,
    n_paths: int = 40_000,
    seed: int = 3,
) -> dict:
    grid = grid or Grid()
    spots, inst_var, variance = simulate_heston_euler(grid, params, n_paths=n_paths, seed=seed)

    spx_model = []
    for q in book.spx:
        st = spots[:, q.expiry_days]
        payoff = np.maximum(st - q.strike, 0.0) if q.call else np.maximum(q.strike - st, 0.0)
        px = float(np.exp(-book.rate * q.tau) * payoff.mean())
        iv = implied_vol(px, book.spot, q.strike, q.tau, book.rate, q.call)
        spx_model.append(
            {**asdict(q), "model_iv": float(iv) if np.isfinite(iv) else None, "model_price": px}
        )
    qv_strip = vix_strip(inst_var, grid)

    fut_model = []
    for q in book.vix_futures:
        # Affine CIR VIX (true conditional) and pathwise QV proxy (apples-to-apples with NN).
        v_t = variance[:, q.obs_days]
        cir_vix = np.sqrt(np.maximum(cir_forward_variance(v_t, params, grid.vix_window_days * grid.dt), 1e-16))
        qv_vix = qv_strip[q.obs_days]
        fut_model.append(
            {
                **asdict(q),
                "model_level_cir": float(cir_vix.mean()),
                "model_level": float(qv_vix.mean()),
            }
        )

    vix_opt_model = []
    for q in book.vix_options:
        vp = qv_strip[q.expiry_days]
        fut = float(vp.mean())
        payoff = np.maximum(vp - q.strike, 0.0) if q.call else np.maximum(q.strike - vp, 0.0)
        px = float(np.exp(-book.rate * q.tau) * payoff.mean())
        iv = implied_vol(px, fut, q.strike, q.tau, book.rate, q.call)
        vix_opt_model.append(
            {
                **asdict(q),
                "model_iv": float(iv) if np.isfinite(iv) else None,
                "model_price": px,
                "model_future": fut,
            }
        )

    def rmse(pairs):
        xs = [a - b for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        return float(np.sqrt(np.mean(np.square(xs)))) if xs else float("nan")

    spx_rmse = rmse([(r["model_iv"], r["iv"]) for r in spx_model if r["model_iv"] is not None])
    fut_rmse = rmse([(r["model_level"], r["level"]) for r in fut_model])
    vix_rmse = rmse([(r["model_iv"], r["iv"]) for r in vix_opt_model if r["model_iv"] is not None])

    return {
        "params": asdict(params),
        "spx": spx_model,
        "vix_futures": fut_model,
        "vix_options": vix_opt_model,
        "rmse": {"spx_iv": spx_rmse, "vix_fut": fut_rmse, "vix_iv": vix_rmse},
        "spots_mean": float(spots[:, -1].mean()),
    }
