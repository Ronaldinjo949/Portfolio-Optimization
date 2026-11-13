"""Einstiegspunkt: führt die komplette Analyse aus und schreibt Grafiken/Tabellen nach results/.

Beispiele:
    python run_analysis.py                  # echte Daten via yfinance
    python run_analysis.py --synthetic      # Offline-Demo mit simulierten Daten
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio import estimators as est
from portfolio import optimizers as opt
from portfolio import plotting as plot
from portfolio.backtest import run_backtest
from portfolio.data import BENCHMARK_TICKER, DEFAULT_UNIVERSE, load_prices, synthetic_prices, to_returns
from portfolio.metrics import summary
from portfolio.strategies import (
    equal_weight_strategy,
    max_sharpe_strategy,
    min_variance_strategy,
    risk_parity_strategy,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--rf", type=float, default=0.02, help="Risikofreier Zins p.a.")
    p.add_argument("--lookback", type=int, default=252, help="Schätzfenster in Handelstagen")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    stock_tickers = list(DEFAULT_UNIVERSE)
    benchmark_ticker = BENCHMARK_TICKER
    all_tickers = stock_tickers + ([benchmark_ticker] if benchmark_ticker not in stock_tickers else [])
    prices = synthetic_prices(all_tickers) if a.synthetic else load_prices(all_tickers, a.start, a.end)
    rets = to_returns(prices)
    stock_rets = rets[stock_tickers]
    print(f"Daten: {stock_rets.index[0].date()} bis {stock_rets.index[-1].date()}, {stock_rets.shape[1]} Aktien")

    # ---- 1) Statische Analyse (In-Sample, nur zur Veranschaulichung) -------------------
    cov, delta = est.ledoit_wolf_cov(stock_rets)
    mu = est.mean_returns(stock_rets)
    print(f"Ledoit-Wolf-Shrinkage-Intensität: {delta:.3f}")

    w_mv = opt.min_variance(cov, a.max_weight)
    w_ms = opt.max_sharpe(mu, cov, a.rf, a.max_weight)
    w_rp = opt.risk_parity(cov)
    vols, fr, _ = opt.efficient_frontier(mu, cov, max_weight=a.max_weight)

    special = {
        "Min-Varianz": (opt.portfolio_vol(w_mv, cov), opt.portfolio_return(w_mv, mu)),
        "Max-Sharpe": (opt.portfolio_vol(w_ms, cov), opt.portfolio_return(w_ms, mu)),
        "Risk Parity": (opt.portfolio_vol(w_rp, cov), opt.portfolio_return(w_rp, mu)),
    }

    frontier_labels = list(stock_rets.columns)
    frontier_vols = np.sqrt(np.diag(cov))
    frontier_rets = mu
    if benchmark_ticker in rets:
        spy = rets[benchmark_ticker]
        frontier_labels.append("SPY")
        frontier_vols = np.append(frontier_vols, float(np.sqrt(252) * spy.std(ddof=0)))
        frontier_rets = np.append(frontier_rets, float(252 * spy.mean()))

    plot.plot_frontier(vols, fr, frontier_vols, frontier_rets, frontier_labels, special, out / "efficient_frontier.png")
    plot.plot_correlation(stock_rets.corr(), out / "correlation.png")
    ws = {"Min-Varianz": w_mv, "Max-Sharpe": w_ms, "Risk Parity": w_rp}
    plot.plot_risk_contributions(
        ws, {k: opt.risk_contributions(v, cov) for k, v in ws.items()}, stock_rets.columns, out / "risk_contributions.png"
    )

    # ---- 2) Walk-forward-Backtest (Out-of-Sample) --------------------------------------
    strategies = {
        "Equal Weight": equal_weight_strategy(),
        "Min-Varianz": min_variance_strategy(a.max_weight),
        "Max-Sharpe": max_sharpe_strategy(a.rf, a.max_weight),
        "Risk Parity": risk_parity_strategy(),
    }
    results = {n: run_backtest(stock_rets, s, a.lookback, a.cost_bps) for n, s in strategies.items()}
    bt = pd.DataFrame({n: r.returns for n, r in results.items()})
    if benchmark_ticker in rets:
        bt[f"{benchmark_ticker} (Buy&Hold)"] = rets[benchmark_ticker].reindex(bt.index)

    table = summary(bt, a.rf)
    table["Ø Turnover/Rebal."] = pd.Series({n: r.turnover.mean() for n, r in results.items()})
    out.mkdir(exist_ok=True)
    table.to_csv(out / "backtest_summary.csv")
    print("\n", table.round(3).to_string())

    plot.plot_equity_curves(bt, out / "equity_curves.png")
    plot.plot_drawdowns(bt, out / "drawdowns.png")
    plot.plot_weights(results["Risk Parity"].weights, "Risk Parity: Gewichte im Zeitverlauf", out / "weights_risk_parity.png")
    plot.plot_weights(results["Max-Sharpe"].weights, "Max-Sharpe: Gewichte im Zeitverlauf", out / "weights_max_sharpe.png")
    print(f"\nErgebnisse gespeichert in: {out.resolve()}")


if __name__ == "__main__":
    main()
