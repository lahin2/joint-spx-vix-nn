"""Vol-desk Streamlit app: joint SPX/VIX neural calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from spx_vix_nn.calibration import price_book_neural, save_checkpoint, train
from spx_vix_nn.config import Grid, TrainConfig
from spx_vix_nn.market import MarketBook
from spx_vix_nn.models.neural_sde import NeuralSLV

ART = ROOT / "artifacts"
PORT_HINT = 43173
VEGA_NOTIONAL = 10_000  # 10k vega book, dollars per vol point


def _load_json(name: str):
    path = ART / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_static():
    book = MarketBook.from_dict(_load_json("book.json"))
    heston = _load_json("heston.json")
    neural = _load_json("neural_mark.json")
    history = _load_json("history.json") or []
    return book, heston, neural, history


def load_model() -> NeuralSLV:
    ckpt = torch.load(ART / "neural_sde.pt", map_location="cpu", weights_only=False)
    model = NeuralSLV(hidden=32, rate=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def smile_figure(title: str, quotes: list, model_key: str = "model_iv") -> go.Figure:
    fig = go.Figure()
    by_t: dict[int, list] = {}
    for q in quotes:
        by_t.setdefault(q["expiry_days"], []).append(q)
    for day, qs in sorted(by_t.items()):
        qs = sorted(qs, key=lambda r: r["log_moneyness"])
        fig.add_trace(
            go.Scatter(
                x=[r["log_moneyness"] for r in qs],
                y=[100 * r["iv"] for r in qs],
                mode="markers+lines",
                name=f"Market {day}d",
                line=dict(width=1),
            )
        )
        if qs[0].get(model_key) is not None:
            fig.add_trace(
                go.Scatter(
                    x=[r["log_moneyness"] for r in qs],
                    y=[100 * r[model_key] if r.get(model_key) is not None else None for r in qs],
                    mode="lines",
                    name=f"Model {day}d",
                    line=dict(dash="dash", width=2),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Log-moneyness  ln(K/S)  or  ln(K/F_VIX)",
        yaxis_title="Implied vol (%)",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
        height=420,
        hovermode="x unified",
    )
    return fig


def futures_figure(heston, neural) -> go.Figure:
    days = [q["obs_days"] for q in heston["vix_futures"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=[100 * q["level"] for q in heston["vix_futures"]], name="Market", mode="markers+lines"))
    fig.add_trace(
        go.Scatter(
            x=days,
            y=[100 * q["model_level"] for q in heston["vix_futures"]],
            name="Heston (SPX-only)",
            mode="lines+markers",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=[100 * q["model_level"] for q in neural["vix_futures"]],
            name="Neural SLV (joint)",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="VIX futures strip (QV proxy, in vol points)",
        xaxis_title="Observation (trading days)",
        yaxis_title="VIX future × 100",
        height=380,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=70),
    )
    return fig


def residual_heatmap(quotes: list, title: str) -> go.Figure:
    days = sorted({q["expiry_days"] for q in quotes})
    ks = sorted({round(q["log_moneyness"], 5) for q in quotes})
    z = []
    for d in days:
        row = []
        for k in ks:
            hit = [q for q in quotes if q["expiry_days"] == d and abs(q["log_moneyness"] - k) < 1e-8]
            if not hit or hit[0].get("model_iv") is None:
                row.append(None)
            else:
                row.append(100 * (hit[0]["model_iv"] - hit[0]["iv"]))
        z.append(row)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{k:.2f}" for k in ks],
            y=[f"{d}d" for d in days],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Model − mkt (vol pts)"),
        )
    )
    fig.update_layout(title=title, xaxis_title="Log-moneyness", height=320, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def loss_figure(history: list) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No training history yet", height=320)
        return fig
    steps = [r["step"] for r in history]
    for key, name in [("loss", "Joint"), ("loss_spx", "SPX IV"), ("loss_fut", "VIX futures"), ("loss_vix", "VIX IV")]:
        fig.add_trace(go.Scatter(x=steps, y=[r[key] for r in history], name=name))
    fig.update_layout(title="Adam joint loss (IV / futures space)", xaxis_title="Step", yaxis_title="Weighted MSE", height=380)
    return fig


def variance_band(neural) -> go.Figure:
    v = np.array(neural["v_mean"])
    lo = np.array(neural["v_q10"])
    hi = np.array(neural["v_q90"])
    t = np.arange(len(v))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(t) + list(t[::-1]), y=list(hi) + list(lo[::-1]), fill="toself", name="10–90% band", line=dict(width=0), opacity=0.35))
    fig.add_trace(go.Scatter(x=t, y=v, name="E[V_t]", mode="lines"))
    fig.update_layout(title="Learned variance factor  V_t", xaxis_title="Trading day", yaxis_title="Variance", height=340)
    return fig


def structure_pnl(heston, neural, book: MarketBook) -> dict:
    """1m variance swap vs VIX future, and a 21d 90-110 VIX risk reversal vs SPX put fly."""
    # Variance swap (continuous) ≈ expected QV / Δ over 21d from t=0, in vol points.
    # Market 21d VIX future is already that object under our proxy.
    mkt_vix_21 = next(q.level for q in book.vix_futures if q.obs_days == 21)
    h_vix_21 = next(q["model_level"] for q in heston["vix_futures"] if q["obs_days"] == 21)
    n_vix_21 = next(q["model_level"] for q in neural["vix_futures"] if q["obs_days"] == 21)

    def vs_price(vol):
        return vol  # already in RMS vol

    # SPX 21d 90/100/110-ish fly using nearest quotes
    def fly_value(quotes, key_iv):
        q21 = [q for q in quotes if q["expiry_days"] == 21]
        if len(q21) < 3:
            return None
        q21 = sorted(q21, key=lambda r: r["log_moneyness"])
        wings = [q21[0], q21[len(q21) // 2], q21[-1]]
        px = []
        for q in wings:
            iv = q[key_iv] if key_iv != "iv" else q["iv"]
            if iv is None:
                return None
            from spx_vix_nn.bs import black_scholes_price

            px.append(float(black_scholes_price(book.spot, q["strike"], q["tau"], iv, book.rate, q["call"])))
        return px[0] - 2 * px[1] + px[2]

    h_fly = fly_value(heston["spx"], "model_iv")
    n_fly = fly_value(neural["spx"], "model_iv")
    m_fly = fly_value([{**asdict_quote(q)} for q in book.spx], "iv")

    # 10k vega on the 21d VIX future: $10k per vol point.
    vix_basis_h = (h_vix_21 - mkt_vix_21) * 100 * VEGA_NOTIONAL
    vix_basis_n = (n_vix_21 - mkt_vix_21) * 100 * VEGA_NOTIONAL
    return {
        "mkt_vix_21": mkt_vix_21,
        "h_vix_21": h_vix_21,
        "n_vix_21": n_vix_21,
        "vs_mkt": vs_price(mkt_vix_21),
        "model_risk_vix_heston": vix_basis_h,
        "model_risk_vix_nn": vix_basis_n,
        "fly_mkt": m_fly,
        "fly_h": h_fly,
        "fly_n": n_fly,
    }


def asdict_quote(q):
    return {
        "expiry_days": q.expiry_days,
        "tau": q.tau,
        "strike": q.strike,
        "log_moneyness": q.log_moneyness,
        "iv": q.iv,
        "bid_ask": q.bid_ask,
        "call": q.call,
        "price": q.price,
    }


def main() -> None:
    st.set_page_config(page_title="Joint SPX/VIX neural calibration", layout="wide")
    st.title("Joint SPX / VIX calibration via neural SDEs")
    st.caption(
        "Guyon–Mustapha programme, reduced to a daily Euler grid: a one-factor Markov "
        "stochastic local-vol model with neural leverage, drift and vol-of-vol, fit to a "
        "synthetic path-dependent-vol book. No live CBOE feed."
    )

    with st.sidebar:
        st.header("How to read this desk")
        st.markdown(
            "You are looking at **one synthetic close**. A path-dependent-vol model minted "
            "the SPX smiles, VIX futures, and VIX options. Two models then try to fit that book."
        )
        st.markdown(
            """
**Heston (SPX-only)** is calibrated to SPX implied vols, then frozen.
It is allowed to miss VIX. That miss *is* the puzzle: vol-of-vol that
builds short-dated SPX skew overstates VIX convexity.

**Neural SLV (joint)** is a one-factor Markov SDE
`σ = L(t, log S) √V` whose drift and diffusion of `V` are small nets
on a Heston backbone. It is scored on SPX IVs + VIX futures + VIX IVs
together.

RMSE is in **vol points** (100 × Black vol). A 2-vol miss on a VIX
future × $10k vega is $20k of model P&L before bid–ask.
"""
        )
        st.markdown("[Full derivation and FAQ](https://github.com/lahin2/joint-spx-vix-nn/blob/main/docs/EXPLANATION.md)")
        st.divider()
        st.caption("VIX here is a 21-day pathwise RMS of instantaneous variance, not the CBOE log-contract strip. Market and model use the same proxy.")

    if not (ART / "book.json").exists():
        st.error("Artifacts missing. Run `python scripts/train_checkpoint.py` first.")
        return

    book, heston, neural, history = load_static()

    tab_mkt, tab_cal, tab_risk = st.tabs(["Market & puzzle", "Calibration", "Risk / structuring"])

    with tab_mkt:
        st.subheader("The joint-calibration puzzle")
        st.markdown(
            "A one-factor **Heston** model is fit to **SPX smiles only**, then frozen and "
            "used to mark VIX futures and VIX options. The same book is then marked with a "
            "**neural SLV** trained on the joint loss. Heston typically keeps SPX residuals "
            "tolerable while missing the VIX strip — the vol-of-vol that manufactures short-dated "
            "SPX skew overstates VIX convexity. Path-dependent vol (the truth model) does not."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Heston SPX IV RMSE", f"{100*heston['rmse']['spx_iv']:.2f} vol pts")
        c2.metric("Heston VIX future RMSE", f"{100*heston['rmse']['vix_fut']:.2f} vol pts")
        c3.metric("Heston VIX IV RMSE", f"{100*heston['rmse']['vix_iv']:.2f} vol pts")
        d1, d2, d3 = st.columns(3)
        d1.metric("Neural SPX IV RMSE", f"{100*neural['rmse']['spx_iv']:.2f} vol pts")
        d2.metric("Neural VIX future RMSE", f"{100*neural['rmse']['vix_fut']:.2f} vol pts")
        d3.metric("Neural VIX IV RMSE", f"{100*neural['rmse']['vix_iv']:.2f} vol pts")
        st.plotly_chart(smile_figure("SPX smiles — Heston (SPX-only fit)", heston["spx"]), use_container_width=True)
        st.plotly_chart(smile_figure("SPX smiles — neural SLV (joint fit)", neural["spx"]), use_container_width=True)
        st.plotly_chart(futures_figure(heston, neural), use_container_width=True)
        st.plotly_chart(smile_figure("21-day VIX smile — Heston", heston["vix_options"]), use_container_width=True)
        st.plotly_chart(smile_figure("21-day VIX smile — neural SLV", neural["vix_options"]), use_container_width=True)
        with st.expander("What a two-vol residual means on a 10k vega book"):
            st.markdown(
                "A **2 vol-point** miss on a VIX future, times a **$10,000 vega** notionals, "
                "is **$20,000** of mark-to-model P&L before any bid–ask. That is why the "
                "futures term in the joint loss is up-weighted: the strip is a tight quote."
            )

    with tab_cal:
        st.subheader("Joint implied-vol loss")
        st.markdown(
            r"The objective is "
            r"$\mathcal{L} = w_{\mathrm{SPX}}\|\Sigma^{\mathrm{SPX}}_{\mathrm{mod}}-\Sigma^{\mathrm{mkt}}\|^2"
            r" + w_{\mathrm{fut}}\|\mathrm{VIX}^{\mathrm{fut}}_{\mathrm{mod}}-\mathrm{VIX}^{\mathrm{fut}}_{\mathrm{mkt}}\|^2"
            r" + w_{\mathrm{VIX}}\|\Sigma^{\mathrm{VIX}}_{\mathrm{mod}}-\Sigma^{\mathrm{mkt}}\|^2$, "
            "with inverse bid–ask weights. Common random numbers and antithetics stay frozen across Adam steps."
        )
        st.plotly_chart(loss_figure(history), use_container_width=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        p = neural["params"]
        p1.metric("v0", f"{p['v0']:.4f}")
        p2.metric("κ", f"{p['kappa']:.3f}")
        p3.metric("θ", f"{p['theta']:.4f}")
        p4.metric("ξ", f"{p['xi']:.3f}")
        p5.metric("ρ", f"{p['rho']:.3f}")
        st.plotly_chart(variance_band(neural), use_container_width=True)
        st.metric("corr(V_t, EWMA of past log-returns)", f"{neural.get('pdv_corr', float('nan')):.3f}")
        st.caption(
            "A negative correlation is the path-dependent-vol fingerprint: the network is allowed "
            "to look at (t, S, V), and after a joint VIX fit the variance factor typically tracks "
            "past spot moves rather than an autonomous CIR shock."
        )

        st.markdown("#### Continue training from the checkpoint")
        st.caption(
            "Short CPU fine-tune with the saved common-random-number seed. Forty steps is a "
            "couple of minutes; the stored checkpoint is already jointly calibrated."
        )
        col_a, col_b, col_c = st.columns(3)
        w_spx = col_a.number_input("w_SPX", 0.1, 10.0, 1.0, 0.1)
        w_fut = col_b.number_input("w_futures", 0.1, 20.0, 6.0, 0.5, help="Up-weight because VIX futures are the tightest quotes in the book.")
        w_vix = col_c.number_input("w_VIX", 0.1, 10.0, 2.0, 0.1)
        steps = st.slider("Adam steps", 5, 40, 10)
        if st.button("Run extra Adam steps", type="primary"):
            with st.spinner("Differentiating through the Euler scheme…"):
                try:
                    model = load_model()
                    cfg = TrainConfig(n_paths=2048, steps=int(steps), lr=8e-3, w_spx=float(w_spx), w_fut=float(w_fut), w_vix=float(w_vix), seed=7)
                    model, hist = train(book, Grid(), cfg, model=model)
                    mark = price_book_neural(model, book, Grid(), n_paths=4096, seed=7)
                    save_checkpoint(ART / "neural_sde.pt", model, (history or []) + hist, extras={"nn_mark": mark["rmse"]})
                    (ART / "neural_mark.json").write_text(json.dumps(mark, indent=2, default=float))
                    (ART / "history.json").write_text(json.dumps((history or []) + hist, indent=2))
                    st.success(f"Updated checkpoint. Neural RMSE {mark['rmse']}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    with tab_risk:
        st.subheader("Residual risk and a sample structure")
        st.markdown(
            "Risk wants the **basis** between SPX-implied variance and VIX-implied variance "
            "after each model is marked. Structuring wants to know which exotic (a VIX call "
            "financed by an SPX put fly, a 1m variance swap vs the VIX future) still has a "
            "model-dependent mid."
        )
        if neural["rmse"]["spx_iv"] != neural["rmse"]["spx_iv"]:
            st.warning("Model IVs were too noisy to invert. Increase MC paths and re-run the checkpoint.")
        else:
            st.plotly_chart(residual_heatmap(heston["spx"], "Heston SPX residual (model − market, vol pts)"), use_container_width=True)
            st.plotly_chart(residual_heatmap(neural["spx"], "Neural SLV SPX residual (model − market, vol pts)"), use_container_width=True)

        pnl = structure_pnl(heston, neural, book)
        st.markdown("##### 21-day VIX future vs 1m variance (QV proxy)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Market 21d VIX", f"{100*pnl['mkt_vix_21']:.2f}")
        m2.metric("Heston mark", f"{100*pnl['h_vix_21']:.2f}")
        m3.metric("Neural mark", f"{100*pnl['n_vix_21']:.2f}")
        st.caption(
            f"Model-risk P&L vs market on a $10k vega VIX future: "
            f"Heston ${pnl['model_risk_vix_heston']:,.0f}, "
            f"neural ${pnl['model_risk_vix_nn']:,.0f} "
            f"(dollar difference = vol-point miss × 100 × $10,000)."
        )
        st.markdown("##### 21-day SPX wing fly (OTM put − 2× ATM + OTM call), Black marks from each smile")
        f1, f2, f3 = st.columns(3)
        f1.metric("Market fly", "—" if pnl["fly_mkt"] is None else f"{pnl['fly_mkt']:.5f}")
        f2.metric("Heston fly", "—" if pnl["fly_h"] is None else f"{pnl['fly_h']:.5f}")
        f3.metric("Neural fly", "—" if pnl["fly_n"] is None else f"{pnl['fly_n']:.5f}")
        if pnl["fly_mkt"] is not None and pnl["fly_h"] is not None and pnl["fly_n"] is not None:
            st.info(
                f"Model-risk P&L on one fly unit: Heston ${10000*(pnl['fly_h']-pnl['fly_mkt']):,.2f}, "
                f"neural ${10000*(pnl['fly_n']-pnl['fly_mkt']):,.2f} (scaled ×10,000 notional)."
            )
        st.caption(
            "Empty cells in the heatmap are quotes where implied vol failed to invert (deep "
            "OTM MC noise). That is a data-quality flag, not a green residual."
        )


if __name__ == "__main__":
    main()
