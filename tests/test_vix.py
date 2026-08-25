import numpy as np

from spx_vix_nn.config import Grid
from spx_vix_nn.vix import vix_pathwise_np


def test_black_scholes_vix_equals_sigma():
    """Constant instantaneous vol implies pathwise VIX proxy equals that vol."""
    grid = Grid()
    n = 512
    sigma = 0.2
    inst = np.full((n, grid.n_steps), sigma**2)
    vix = vix_pathwise_np(inst, obs_idx=0, window=grid.vix_window_days)
    assert np.allclose(vix, sigma, atol=1e-12)
