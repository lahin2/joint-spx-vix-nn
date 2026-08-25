"""Joint implied-vol loss and Adam loop for the neural SLV."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.functional import relu

from .bs import black_scholes_price, implied_vol
from .config import Grid, TrainConfig
from .market import MarketBook
from .models.neural_sde import NeuralSLV, make_noise
from .vix import vix_pathwise_torch


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x.square()) / np.sqrt(2.0 * np.pi)


def torch_bs_price(
    spot: torch.Tensor,
    strike: torch.Tensor,
    tau: torch.Tensor,
    vol: torch.Tensor,
    rate: float,
    call: torch.Tensor,
) -> torch.Tensor:
    vol = vol.clamp(min=1e-6)
    tau = tau.clamp(min=1e-8)
    fwd = spot * torch.exp(torch.as_tensor(rate, dtype=vol.dtype, device=vol.device) * tau)
    m = torch.log(fwd.clamp(min=1e-12) / strike.clamp(min=1e-12))
    sig_sqrt = vol * torch.sqrt(tau)
    d1 = m / sig_sqrt + 0.5 * sig_sqrt
    d2 = d1 - sig_sqrt
    df = torch.exp(-rate * tau)
    call_px = df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    put_px = df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))
    return torch.where(call.bool(), call_px, put_px)


def torch_implied_vol(
    price: torch.Tensor,
    spot: torch.Tensor,
    strike: torch.Tensor,
    tau: torch.Tensor,
    rate: float,
    call: torch.Tensor,
    iters: int = 8,
) -> torch.Tensor:
    seed = torch.sqrt(2.0 * np.pi / tau.clamp(min=1e-8)) * price.clamp(min=1e-8) / spot.clamp(min=1e-8)
    vol = seed.clamp(0.03, 1.5)
    for _ in range(iters):
        px = torch_bs_price(spot, strike, tau, vol, rate, call)
        fwd = spot * torch.exp(torch.as_tensor(rate, dtype=vol.dtype, device=vol.device) * tau)
        m = torch.log(fwd.clamp(min=1e-12) / strike.clamp(min=1e-12))
        sig_sqrt = vol.clamp(min=1e-6) * torch.sqrt(tau.clamp(min=1e-8))
        d1 = m / sig_sqrt + 0.5 * sig_sqrt
        df = torch.exp(-rate * tau)
        vega = df * fwd * torch.sqrt(tau.clamp(min=1e-8)) * _norm_pdf(d1)
        vol = (vol - (px - price) / vega.clamp(min=1e-6)).clamp(1e-4, 4.0)
    return vol


def mc_option_prices(
    spots_at_t: torch.Tensor,
    strikes: torch.Tensor,
    call: torch.Tensor,
    df: float,
) -> torch.Tensor:
    s = spots_at_t.unsqueeze(1)
    k = strikes.unsqueeze(0)
    payoff = torch.where(call.unsqueeze(0), relu(s - k), relu(k - s))
    return df * payoff.mean(dim=0)


def book_tensors(book: MarketBook, device: str) -> dict[str, Any]:
    def stack_vanillas(qs, spot_for_iv: float | None = None):
        return {
            "expiry_days": torch.tensor([q.expiry_days for q in qs], device=device),
            "tau": torch.tensor([q.tau for q in qs], dtype=torch.float32, device=device),
            "strike": torch.tensor([q.strike for q in qs], dtype=torch.float32, device=device),
            "iv": torch.tensor([q.iv for q in qs], dtype=torch.float32, device=device),
            "w": torch.tensor([1.0 / max(q.bid_ask, 1e-4) for q in qs], dtype=torch.float32, device=device),
            "call": torch.tensor([q.call for q in qs], device=device),
            "log_m": torch.tensor([q.log_moneyness for q in qs], dtype=torch.float32, device=device),
        }

    return {
        "spx": stack_vanillas(book.spx),
        "vix_opt": stack_vanillas(book.vix_options),
        "fut_days": torch.tensor([q.obs_days for q in book.vix_futures], device=device),
        "fut_level": torch.tensor([q.level for q in book.vix_futures], dtype=torch.float32, device=device),
        "fut_w": torch.tensor([1.0 / max(q.bid_ask, 1e-4) for q in book.vix_futures], dtype=torch.float32, device=device),
        "spot": book.spot,
        "rate": book.rate,
    }


def joint_loss(
    model: NeuralSLV,
    book: MarketBook,
    grid: Grid,
    noise: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
) -> dict[str, torch.Tensor]:
    device = noise[0].device
    tbook = book_tensors(book, str(device))
    spots, inst_var, v_paths, log_s = model.simulate(noise[0], noise[1], grid)

    # SPX vanillas grouped by expiry
    spx_iv_err = []
    spx_w = []
    for day in grid.spx_expiry_days:
        mask = tbook["spx"]["expiry_days"] == day
        if not torch.any(mask):
            continue
        tau = tbook["spx"]["tau"][mask]
        k = tbook["spx"]["strike"][mask]
        call = tbook["spx"]["call"][mask]
        mkt = tbook["spx"]["iv"][mask]
        w = tbook["spx"]["w"][mask]
        df = float(np.exp(-book.rate * float(tau[0])))
        px = mc_option_prices(spots[:, day], k, call, df)
        spot_t = torch.full_like(k, book.spot)
        iv = torch_implied_vol(px, spot_t, k, tau, book.rate, call)
        spx_iv_err.append((iv - mkt) * torch.sqrt(w))
        spx_w.append(w)

    spx_vec = torch.cat(spx_iv_err) if spx_iv_err else torch.zeros(1, device=device)
    loss_spx = spx_vec.square().mean()

    fut_err = []
    vix_by_day: dict[int, torch.Tensor] = {}
    for day in grid.vix_obs_days:
        vix_by_day[day] = vix_pathwise_torch(inst_var, day, grid.vix_window_days)
    for i, day in enumerate(grid.vix_obs_days):
        model_fut = vix_by_day[int(day)].mean()
        mkt_fut = tbook["fut_level"][i]
        w = tbook["fut_w"][i]
        fut_err.append(torch.sqrt(w) * (model_fut - mkt_fut))
    loss_fut = torch.stack(fut_err).square().mean() if fut_err else torch.zeros((), device=device)

    vix_err = []
    if book.vix_options:
        # Options on the 21d VIX observation
        day = 21
        vp = vix_by_day[day]
        fut = vp.mean()
        k = tbook["vix_opt"]["strike"]
        call = tbook["vix_opt"]["call"]
        tau = tbook["vix_opt"]["tau"]
        mkt = tbook["vix_opt"]["iv"]
        w = tbook["vix_opt"]["w"]
        df = float(np.exp(-book.rate * float(tau[0])))
        px = mc_option_prices(vp, k, call, df)
        iv = torch_implied_vol(px, fut.expand_as(k), k, tau, book.rate, call)
        vix_err.append((iv - mkt) * torch.sqrt(w))
    loss_vix = torch.cat(vix_err).square().mean() if vix_err else torch.zeros((), device=device)

    total = cfg.w_spx * loss_spx + cfg.w_fut * loss_fut + cfg.w_vix * loss_vix
    return {
        "loss": total,
        "loss_spx": loss_spx,
        "loss_fut": loss_fut,
        "loss_vix": loss_vix,
        "spots": spots,
        "inst_var": inst_var,
        "v_paths": v_paths,
        "log_s": log_s,
    }


@torch.no_grad()
def price_book_neural(
    model: NeuralSLV,
    book: MarketBook,
    grid: Grid,
    n_paths: int = 8192,
    seed: int = 7,
    antithetic: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    w, z = make_noise(n_paths, grid.n_steps, seed, antithetic, device)
    spots, inst_var, v_paths, log_s = model.simulate(w, z, grid)
    s_np = spots.cpu().numpy()
    ivar = inst_var.cpu().numpy()
    v_np = v_paths.cpu().numpy()
    log_np = log_s.cpu().numpy()

    spx_out = []
    for q in book.spx:
        st = s_np[:, q.expiry_days]
        payoff = np.maximum(st - q.strike, 0.0) if q.call else np.maximum(q.strike - st, 0.0)
        px = float(np.exp(-book.rate * q.tau) * payoff.mean())
        iv = implied_vol(px, book.spot, q.strike, q.tau, book.rate, q.call)
        spx_out.append({**asdict(q), "model_iv": float(iv) if np.isfinite(iv) else None, "model_price": px})

    from .vix import vix_strip

    strip = vix_strip(ivar, grid)
    fut_out = []
    for q in book.vix_futures:
        fut_out.append({**asdict(q), "model_level": float(strip[q.obs_days].mean())})

    vix_out = []
    for q in book.vix_options:
        vp = strip[q.expiry_days]
        fut = float(vp.mean())
        payoff = np.maximum(vp - q.strike, 0.0) if q.call else np.maximum(q.strike - vp, 0.0)
        px = float(np.exp(-book.rate * q.tau) * payoff.mean())
        iv = implied_vol(px, fut, q.strike, q.tau, book.rate, q.call)
        vix_out.append(
            {
                **asdict(q),
                "model_iv": float(iv) if np.isfinite(iv) else None,
                "model_price": px,
                "model_future": fut,
            }
        )

    def rmse(pairs):
        xs = [a - b for a, b in pairs if a is not None and np.isfinite(a) and np.isfinite(b)]
        return float(np.sqrt(np.mean(np.square(xs)))) if xs else float("nan")

    # PDV fingerprint: corr(V_t, EWMA of past log-returns)
    rets = np.diff(log_np, axis=1)
    ewma = np.zeros_like(v_np)
    decay = np.exp(-8.0 * grid.dt)
    acc = np.zeros(v_np.shape[0])
    for i in range(1, v_np.shape[1]):
        acc = decay * acc + (1 - decay) * rets[:, i - 1]
        ewma[:, i] = acc
    mid = v_np.shape[1] // 2
    corr = float(np.corrcoef(v_np[:, mid], ewma[:, mid])[0, 1])

    return {
        "params": model.parameter_dict(),
        "spx": spx_out,
        "vix_futures": fut_out,
        "vix_options": vix_out,
        "rmse": {
            "spx_iv": rmse([(r["model_iv"], r["iv"]) for r in spx_out]),
            "vix_fut": rmse([(r["model_level"], r["level"]) for r in fut_out]),
            "vix_iv": rmse([(r["model_iv"], r["iv"]) for r in vix_out]),
        },
        "pdv_corr": corr,
        "v_mean": v_np.mean(axis=0).tolist(),
        "v_q10": np.quantile(v_np, 0.1, axis=0).tolist(),
        "v_q90": np.quantile(v_np, 0.9, axis=0).tolist(),
        "spot_mean": float(s_np[:, -1].mean()),
    }


def train(
    book: MarketBook,
    grid: Grid | None = None,
    cfg: TrainConfig | None = None,
    model: NeuralSLV | None = None,
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[NeuralSLV, list[dict[str, float]]]:
    grid = grid or Grid()
    cfg = cfg or TrainConfig()
    model = model or NeuralSLV(hidden=cfg.hidden, rate=book.rate)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    noise = make_noise(cfg.n_paths, grid.n_steps, cfg.seed, cfg.antithetic, cfg.device)
    history: list[dict[str, float]] = []
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        out = joint_loss(model, book, grid, noise, cfg)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        rec = {
            "step": step,
            "loss": float(out["loss"].detach()),
            "loss_spx": float(out["loss_spx"].detach()),
            "loss_fut": float(out["loss_fut"].detach()),
            "loss_vix": float(out["loss_vix"].detach()),
            **model.parameter_dict(),
        }
        history.append(rec)
        if log_fn is not None:
            log_fn(step, rec)
    model.eval()
    return model, history


def save_checkpoint(
    path: Path,
    model: NeuralSLV,
    history: list[dict[str, float]],
    extras: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "history": history,
        "params": model.parameter_dict(),
        "extras": extras or {},
    }
    torch.save(payload, path)


def load_history(path: Path) -> list[dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("history", [])
