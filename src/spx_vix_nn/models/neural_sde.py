"""One-factor Markov neural stochastic local-vol SDE.

    d log S = (r - ½ σ²) dt + σ dW
    σ     = L_θ(t, log S) √V
    dV    = μ_θ(t, log S, V) dt + ν_θ(t, log S, V) dZ
    d⟨W,Z⟩ = ρ dt

Leverage, drift and vol-of-vol are small MLPs sitting on a Heston-like
backbone so a CPU Adam run stays stable. This is the Guyon–Mustapha neural
SDE programme reduced to a daily Euler grid.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import Grid, TrainConfig


class MLP(nn.Module):
    def __init__(self, din: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralSLV(nn.Module):
    def __init__(self, hidden: int = 32, rate: float = 0.0) -> None:
        super().__init__()
        self.rate = rate
        self.leverage = MLP(2, hidden)
        self.mu_res = MLP(3, hidden)
        self.nu_res = MLP(3, hidden)
        self.raw_v0 = nn.Parameter(torch.tensor(0.20))  # sqrt(v0)
        self.raw_kappa = nn.Parameter(torch.tensor(1.2))
        self.raw_theta = nn.Parameter(torch.tensor(0.20))
        self.raw_xi = nn.Parameter(torch.tensor(0.7))
        self.raw_rho = nn.Parameter(torch.tensor(-0.9))  # atanh-ish via tanh

    @property
    def v0(self) -> torch.Tensor:
        return self.raw_v0.square() + 1e-4

    @property
    def kappa(self) -> torch.Tensor:
        return self.raw_kappa.square() + 0.05

    @property
    def theta(self) -> torch.Tensor:
        return self.raw_theta.square() + 1e-4

    @property
    def xi(self) -> torch.Tensor:
        return self.raw_xi.square() + 0.05

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self.raw_rho)

    def features(self, t: torch.Tensor, log_s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t_n = t.expand_as(log_s)
        return torch.stack([t_n, log_s, torch.sqrt(v.clamp(min=1e-8))], dim=-1)

    def sigma(self, t: torch.Tensor, log_s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        lev = 0.65 + torch.nn.functional.softplus(self.leverage(torch.stack([t.expand_as(log_s), log_s], dim=-1)))
        return lev * torch.sqrt(v.clamp(min=1e-8))

    def drift_vol(self, t: torch.Tensor, log_s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(t, log_s, v)
        v_pos = v.clamp(min=1e-8)
        mu = self.kappa * (self.theta - v) + 1.5 * torch.tanh(self.mu_res(feat))
        nu = self.xi * torch.sqrt(v_pos) * (0.55 + torch.nn.functional.softplus(self.nu_res(feat)))
        return mu, nu

    def simulate(
        self,
        noise_w: torch.Tensor,
        noise_z: torch.Tensor,
        grid: Grid,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Euler–Maruyama.

        noise_*: [n_paths, n_steps] standard normals (antithetics already stacked).
        Returns spots, inst_var, V, log_s  with inst_var [n, n_steps], others [n, n_steps+1].
        """
        n_paths, n_steps = noise_w.shape
        device = noise_w.device
        dt = grid.dt
        sqrt_dt = dt**0.5
        rho = self.rho
        z = rho * noise_w + torch.sqrt(torch.clamp(1.0 - rho**2, min=0.0)) * noise_z

        log_s = torch.full((n_paths,), float(np_log(grid.spot)), device=device, dtype=noise_w.dtype)
        v = self.v0 * torch.ones(n_paths, device=device, dtype=noise_w.dtype)

        spots = []
        vs = []
        inst = []
        logs = []
        spots.append(torch.exp(log_s))
        vs.append(v)
        logs.append(log_s)

        for i in range(n_steps):
            t = torch.tensor((i + 0.5) * dt, device=device, dtype=noise_w.dtype)
            sig = self.sigma(t, log_s, v)
            inst.append(sig.square())
            mu, nu = self.drift_vol(t, log_s, v)
            log_s = log_s + (grid.rate - 0.5 * sig.square()) * dt + sig * sqrt_dt * noise_w[:, i]
            v = v + mu * dt + nu * sqrt_dt * z[:, i]
            v = torch.clamp(v, min=1e-6, max=2.0)
            spots.append(torch.exp(log_s))
            vs.append(v)
            logs.append(log_s)

        return (
            torch.stack(spots, dim=1),
            torch.stack(inst, dim=1),
            torch.stack(vs, dim=1),
            torch.stack(logs, dim=1),
        )

    def parameter_dict(self) -> dict[str, float]:
        return {
            "v0": float(self.v0.detach()),
            "kappa": float(self.kappa.detach()),
            "theta": float(self.theta.detach()),
            "xi": float(self.xi.detach()),
            "rho": float(self.rho.detach()),
        }


def np_log(x: float) -> float:
    import math

    return math.log(x)


def make_noise(n_paths: int, n_steps: int, seed: int, antithetic: bool, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    half = n_paths // 2 if antithetic else n_paths
    w = torch.randn(half, n_steps, generator=g)
    z = torch.randn(half, n_steps, generator=g)
    if antithetic:
        w = torch.cat([w, -w], dim=0)
        z = torch.cat([z, -z], dim=0)
    return w.to(device), z.to(device)


def load_model(path: str, cfg: TrainConfig | None = None) -> NeuralSLV:
    cfg = cfg or TrainConfig()
    model = NeuralSLV(hidden=cfg.hidden, rate=0.0)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model
