"""Walk-forward-Backtest ohne Look-ahead-Bias, mit Transaktionskosten und Gewichtsdrift."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series      # Netto-Tagesrenditen (nach Kosten)
    weights: pd.DataFrame   # Zielgewichte zu den Rebalancing-Terminen
    turnover: pd.Series     # Umschichtung je Rebalancing


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    lookback: int = 252,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Monatliches Rebalancing.

    Am ersten Handelstag eines Monats werden die Gewichte nur mit Daten VOR diesem Tag
    geschätzt (returns.iloc[t-lookback:t]) -> kein Look-ahead. Zwischen den Terminen
    driften die Gewichte mit den Renditen. Kosten = Turnover * cost_bps.
    """
    idx = returns.index
    months = pd.Series(idx.to_period("M"), index=idx)
    month_start = months.ne(months.shift()).values
    R = returns.values

    w = np.zeros(R.shape[1])
    first = True
    out, w_hist, to_hist = [], {}, {}

    for t in range(lookback, len(returns)):
        cost = 0.0
        if first or month_start[t]:
            target = strategy(returns.iloc[t - lookback : t])
            turnover = np.abs(target - w).sum()
            cost = turnover * cost_bps / 1e4
            w_hist[idx[t]] = target
            to_hist[idx[t]] = turnover
            w, first = target, False
        gross = float(w @ R[t])
        out.append((idx[t], gross - cost))
        w = w * (1 + R[t]) / (1 + gross)  # Drift

    net = pd.Series(dict(out), name="return")
    return BacktestResult(
        returns=net,
        weights=pd.DataFrame(w_hist, index=returns.columns).T,
        turnover=pd.Series(to_hist, name="turnover"),
    )
