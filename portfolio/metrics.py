"""Performance- und Risikokennzahlen."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    excess = r - rf / TRADING_DAYS
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    excess = r - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def var_cvar(r: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Historischer Tages-VaR und CVaR (Expected Shortfall) als positive Verlustzahlen."""
    q = r.quantile(1 - alpha)
    return float(-q), float(-r[r <= q].mean())


def summary(returns: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        var, cvar = var_cvar(r)
        mdd = max_drawdown(r)
        rows[name] = {
            "CAGR": cagr(r),
            "Volatilität": ann_vol(r),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
            "Max Drawdown": mdd,
            "Calmar": cagr(r) / abs(mdd) if mdd < 0 else np.nan,
            "VaR 95% (1d)": var,
            "CVaR 95% (1d)": cvar,
        }
    return pd.DataFrame(rows).T
