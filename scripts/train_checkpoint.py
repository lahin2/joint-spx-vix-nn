"""Build the synthetic book, fit Heston, train the neural SDE, dump artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spx_vix_nn.calibration import price_book_neural, save_checkpoint, train
from spx_vix_nn.config import Grid, TrainConfig
from spx_vix_nn.market import build_synthetic_book
from spx_vix_nn.models.heston import calibrate_heston_spx, price_book_heston


def main() -> None:
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    grid = Grid()
    print("Building synthetic PDV book...")
    book = build_synthetic_book(grid, n_paths=60_000, seed=11)
    (art / "book.json").write_text(json.dumps(book.to_dict(), indent=2))
    print(f"  SPX quotes={len(book.spx)}  VIX futures={len(book.vix_futures)}  VIX opts={len(book.vix_options)}")
    print("  VIX futures:", [round(q.level, 4) for q in book.vix_futures])

    print("Calibrating Heston to SPX only...")
    heston = calibrate_heston_spx(book, grid, n_paths=10_000, seed=5)
    heston_mark = price_book_heston(book, heston, grid, n_paths=25_000, seed=3)
    (art / "heston.json").write_text(json.dumps(heston_mark, indent=2))
    print("  Heston params:", heston)
    print("  Heston RMSE", heston_mark["rmse"])

    cfg = TrainConfig(n_paths=4096, steps=70, lr=1e-2, seed=7, w_spx=1.0, w_fut=6.0, w_vix=2.0)
    print(f"Training neural SLV for {cfg.steps} Adam steps...")

    def log(step: int, rec: dict) -> None:
        if step % 5 == 0 or step == cfg.steps - 1:
            print(
                f"  step {step:03d}  loss={rec['loss']:.5f}  "
                f"spx={rec['loss_spx']:.5f}  fut={rec['loss_fut']:.5f}  vix={rec['loss_vix']:.5f}  "
                f"rho={rec['rho']:.3f} v0={rec['v0']:.4f}"
            )

    model, history = train(book, grid, cfg, log_fn=log)
    nn_mark = price_book_neural(model, book, grid, n_paths=8192, seed=7)
    print("  Neural RMSE", nn_mark["rmse"], "pdv_corr", nn_mark["pdv_corr"])

    extras = {
        "train_cfg": cfg.__dict__,
        "nn_mark": {k: nn_mark[k] for k in ("params", "rmse", "pdv_corr", "v_mean", "v_q10", "v_q90")},
        "heston_rmse": heston_mark["rmse"],
    }
    save_checkpoint(art / "neural_sde.pt", model, history, extras)
    (art / "neural_mark.json").write_text(
        json.dumps({k: nn_mark[k] for k in nn_mark if k not in ()}, indent=2, default=float)
    )
    (art / "history.json").write_text(json.dumps(history, indent=2))
    print("Wrote", art)


if __name__ == "__main__":
    main()
