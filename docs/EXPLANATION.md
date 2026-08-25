# What this project is doing (desk notes)

Use this file as the screen-share script. The README is the map; this is the conversation.

## 1. The market object

There is no Bloomberg or CBOE feed. The “market” is a Monte Carlo book from a **path-dependent vol** truth model (`src/spx_vix_nn/models/pdv.py`):

- Instantaneous vol is \(\sigma_0\exp(\nu X_t + \ell \log(S_t/S_0))\).
- \(X_t\) is a fast mean-reverting Gaussian factor driven by a Brownian motion **correlated** with spot (the usual equity \(\rho < 0\)).
- The extra \(\ell\log(S/S_0)\) term is **leverage / local vol**: OTM puts see higher vol because the spot has already fallen, even if \(X\) has not jumped.

That combination makes a **steep short-dated SPX put skew** without needing Heston-sized vol-of-vol. The book stores:

| Instrument | Grid | Quote |
| --- | --- | --- |
| SPX vanillas | 5d, 21d, 42d × 8 log-moneyness points | Black implied vol, bid–ask width |
| VIX futures | same three dates | mean of the QV proxy |
| VIX options | 21d observation, 80–120% moneyness vs the future | Black implied vol on that future |

OTM quotes are stored as OTM puts/calls so intrinsic does not pollute implied vol.

## 2. The Heston baseline (the puzzle exhibit)

Heston is fit by least squares **to SPX implied vols only** (`calibrate_heston_spx`), then frozen.

VIX is marked two ways in the JSON (only the QV proxy is compared to the neural SDE):

- **CIR affine map** — the true Heston 30-day expected variance, \(\mathrm{VIX}^2_t = a\,v_t + b\).
- **Pathwise QV** — same Riemann-sum proxy the neural model uses, so the RMSE is apples-to-apples.

What you should see, and what the shipped checkpoint shows:

- SPX smiles: Heston is *tolerable* (~1.3 vol pts RMSE).
- VIX futures: Heston is **cheap** vs the PDV book (~5 vol pts).
- VIX options: Heston **overstates convexity** (model IVs ~280 vol vs market ~108). Autonomous vol-of-vol that was used to fit SPX skew shows up as a fat VIX smile.

That is the joint-calibration puzzle in one slide.

## 3. The neural SDE

One-factor Markov stochastic *local* vol:

\[
\sigma_t = L_\theta(t,\log S_t)\sqrt{V_t}
\]

\(L_\theta\) is a small MLP (features: calendar time and log-spot). Drift and diffusion of \(V\) are a **Heston backbone plus neural residuals**:

\[
\mu = \kappa(\theta-V) + 1.5\tanh(\mathrm{MLP}_\mu),\qquad
\nu = \xi\sqrt{V}\,(0.55 + \mathrm{softplus}(\mathrm{MLP}_\nu))
\]

Why a backbone? A free neural drift on variance will explode or collapse to zero on a 63-step Euler scheme long before it fits smiles. The backbone is the same idea as a physics-informed residual: the network is allowed to *deform* Heston, not replace Brownian motion with an unconstrained MLP.

Simulation: Euler–Maruyama, 4096 antithetic paths, common random numbers held fixed for the whole Adam run (`make_noise` in `neural_sde.py`).

## 4. VIX: three objects people conflate

| Object | Filtration | What it is |
| --- | --- | --- |
| CBOE VIX | \(\mathcal{F}_t\) | Discrete log-contract strip on SPX options with 30-day expiry |
| True model VIX | \(\mathcal{F}_t\) | \(\sqrt{\mathbb{E}[\frac1\Delta\int_t^{t+\Delta}\sigma^2\mid\mathcal{F}_t]}\) |
| **This repo’s proxy** | \(\mathcal{F}_{t+\Delta}\) | Pathwise RMS of \(\sigma^2\) from \(t\) to \(t+\Delta\) |

We use the third because nested Monte Carlo (inner paths from every outer path at each VIX date, through autodiff) is too heavy for a CPU demo. Market *and* model use the same proxy, so calibration is coherent.

Jensen gap: \(\mathbb{E}\sqrt{\mathrm{QV}} \neq \sqrt{\mathbb{E}[\mathrm{QV}]}\) and \(\mathbb{E}\sqrt{\mathrm{QV}} \neq \mathbb{E}\sqrt{\mathbb{E}[\mathrm{QV}\mid\mathcal{F}_t]}\). Say this out loud in a risk interview. It is why a variance swap, a VIX future, and listed VIX options are not the same instrument.

## 5. Why the loss is in implied vol

Price-space MSE over-weights ATM high-vega options and under-weights wings the desk actually cares about. Vega-weighted price MSE is almost IV MSE. We invert Black with a few Newton steps inside the graph (`torch_implied_vol`) so Adam sees vol points, which is how a vol book is P&L’d.

Quotes are weighted by \(1/\text{bid–ask}\). VIX futures are the tightest quotes, so \(w_{\mathrm{fut}}\) defaults to 6.

## 6. The PDV fingerprint

After the joint fit we correlate \(V_t\) at mid-horizon with an EWMA of *past* log-returns. Shipped checkpoint: **−0.78**.

The SDE is Markov in \((S,V)\), so this is not “the model is non-Markov.” It means the *calibrated* factor \(V\) has learned to track the same information a PDV model would put in an EWMA of returns. That is Guyon–Mustapha’s qualitative result, reproduced on a toy book.

## 7. Structures on the risk tab

**21-day VIX future vs 1m variance.** Under the QV proxy they are the same object at \(t=0\). Model-risk P&L is \((\text{model} - \text{market})\times 100 \times \$10{,}000\) vega. Heston is ~$49k off; the neural SLV is ~$12k off on the shipped book.

**21-day SPX wing fly** (lowest strike − 2× mid + highest strike, Black-marked off each smile). This is a crude stand-in for “the put fly you sold to finance a VIX call.” The dollar difference between Heston, neural, and market mids is inventory you are warehousing if you only calibrated one smile.

## 8. Interview FAQ

**Why not rough vol / two-factor Bergomi?** Those are the right *production* answers. This repo’s claim is narrower: a *one-factor Markov neural SLV* can already co-fit the toy book, and the factor looks path-dependent. Adding a second factor would hide the punchline.

**Why not real SPX/VIX snapshots?** Reproducibility. A CSV schema lives on `VanillaQuote` / `VixFutureQuote`; swapping in a live snapshot is a data problem, not a modelling one.

**Is the Heston CF used?** It is implemented (`heston_call`) for reference. The baseline *calibration* uses Euler MC with common random numbers so SPX and VIX live on the same paths.

**Can I trust the 183 vol RMSE on Heston VIX options?** Directionally yes (convexity is wildly overstated). The number is large because Heston VIX IVs land near 280 while the PDV proxy’s “VIX vol” is already ~108 — short-dated options on a pathwise realized-vol proxy are themselves very convex. Do not quote 183 as a CBOE number.

**What would you do next on a desk?** (1) Replace the QV proxy with a discrete log-contract strip. (2) Nested MC or a Markov regression for true \(\mathcal{F}_t\) VIX. (3) Calibrate to a real snapshot with bid–ask filters. (4) Regularize the leverage function against Dupire local vol. (5) Put the residual heatmap on a limit.
