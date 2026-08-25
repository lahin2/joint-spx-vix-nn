from spx_vix_nn.bs import black_scholes_price, implied_vol


def test_implied_vol_round_trip():
    spot, k, tau, vol, rate = 1.0, 0.95, 21 / 252, 0.22, 0.0
    px = black_scholes_price(spot, k, tau, vol, rate, call=True)
    iv = implied_vol(px, spot, k, tau, rate, call=True)
    assert abs(float(iv) - vol) < 1e-6


def test_put_call_parity_round_trip():
    spot, k, tau, vol = 1.0, 1.05, 42 / 252, 0.18
    px = black_scholes_price(spot, k, tau, vol, 0.0, call=False)
    iv = implied_vol(px, spot, k, tau, 0.0, call=False)
    assert abs(float(iv) - vol) < 1e-6
