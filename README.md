# Joint SPX/VIX Calibration via Neural Networks

A desk-interview research demo for **quantitative research (derivatives / volatility)**, **structuring**, and **risk**.

It does three things you can walk through in a screen-share:

1. Mint a synthetic but economically faithful **SPX + VIX book** from a path-dependent volatility (PDV) truth model.
2. Fit a **Heston** model to **SPX smiles only**, then freeze it and mark VIX — this is the classic joint-calibration *puzzle*.
3. Jointly calibrate a **neural stochastic local-vol SDE** (Guyon–Mustapha style) so SPX vanillas, VIX futures, and VIX options are scored in the *same* implied-vol loss.

This is **not** a production calibrator and **not** a CBOE VIX replication. It is a computationally honest slice: daily Euler Monte Carlo, small networks, three maturities, a quadratic-variation VIX proxy.

Shipped CPU checkpoint (same units as a vol desk: *vol points* = 100 × RMS vol):

| Model | SPX IV RMSE | VIX future RMSE | VIX option IV RMSE |
| --- | ---: | ---: | ---: |
| Heston (SPX-only) | 1.31 | 4.80 | 183 |
| Neural SLV (joint) | 0.97 | 1.34 | 1.70 |

The learned variance factor correlates **−0.78** with an EWMA of past log-returns — the path-dependent-vol fingerprint.

## Why this problem exists

Short-dated **SPX put skew** is steep. In a one-factor Heston model the only way to manufacture that skew is **a lot of negatively correlated vol-of-vol**. That same vol-of-vol inflates **VIX futures** and **VIX option convexity**. Path-dependent volatility can create SPX skew from *past returns* without an autonomous CIR shock.

Guyon & Mustapha (*Neural Joint S&P 500/VIX Smile Calibration*, Risk 2023) show a **one-factor Markov SLV** whose drift and diffusion are neural nets can fit SPX smiles, VIX futures, and VIX smiles together.

Read [docs/EXPLANATION.md](docs/EXPLANATION.md) for the derivation, the VIX-proxy caveat, and an interview FAQ.

## Model

Spot S and variance factor V:

    d log S = (r - 0.5 σ²) dt + σ dW
    σ = L_θ(t, log S) √V
    dV = μ_θ(t, log S, V) dt + ν_θ(t, log S, V) dZ

L, μ, ν are tiny MLPs on a Heston backbone. VIX in this repo is the 21-day pathwise RMS of instantaneous variance (not the CBOE log-contract). Market and model use the same proxy.

Loss is weighted MSE in **implied vol** (SPX and VIX options) plus VIX futures, with inverse bid–ask weights and frozen common-random-number Monte Carlo.

## Run locally

```bash
pip install -e ".[dev]"
python scripts/train_checkpoint.py
streamlit run app.py --server.port 43173 --server.address 0.0.0.0
pytest -q
```

Dashboard tabs: **Market & puzzle** (Heston vs neural smiles), **Calibration** (joint loss, PDV fingerprint), **Risk / structuring** ($10k vega model-risk P&L).

A **2 vol-point** miss on a VIX future × **$10,000 vega** is **$20,000** of mark-to-model P&L before bid–ask.

## Layout

- `app.py` — Streamlit vol desk (sidebar explains how to read the charts)
- `docs/EXPLANATION.md` — math, VIX proxy, interview FAQ
- `src/spx_vix_nn/` — PDV book, Heston baseline, neural SLV, joint IV loss
- `artifacts/` — pretrained checkpoint and priced book
