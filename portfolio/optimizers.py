"""Optimierer: Min-Varianz, Max-Sharpe, Effizienzkurve (Markowitz) und Risk Parity."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def _solve(objective, n, extra_constraints=(), max_weight=1.0, x0=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    start = np.asarray(x0 if x0 is not None else np.full(n, 1.0 / n), dtype=float)
    if start.size != n:
        start = np.full(n, 1.0 / n)
    start = np.clip(start, 0.0, max_weight)
    if start.sum() <= 0:
        start = np.full(n, 1.0 / n)
    start = start / start.sum()

    res = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,  # long-only
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        fallback = start.copy()
        fallback = np.clip(fallback, 0.0, max_weight)
        fallback = fallback / fallback.sum()
        if np.all((fallback >= -1e-12) & (fallback <= max_weight + 1e-12)) and abs(fallback.sum() - 1.0) < 1e-8:
            return fallback
        raise RuntimeError(f"Optimierung fehlgeschlagen: {res.message}")
    return res.x


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    return _solve(lambda w: w @ cov @ w, len(cov), max_weight=max_weight)


def max_sharpe(mu, cov, rf: float = 0.0, max_weight: float = 1.0) -> np.ndarray:
    return _solve(
        lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), len(mu), max_weight=max_weight
    )


def min_variance_for_target(mu, cov, target_return: float, max_weight: float = 1.0):
    ret_con = {"type": "eq", "fun": lambda w: w @ mu - target_return}
    return _solve(lambda w: w @ cov @ w, len(mu), [ret_con], max_weight)


def efficient_frontier(mu, cov, n_points: int = 40, max_weight: float = 1.0):
    """Berechnet die Effizienzkurve; gibt (vols, rets, weights) zurück."""
    w_min = min_variance(cov, max_weight)
    lo = portfolio_return(w_min, mu)
    # Maximal erreichbare Rendite unter Gewichtsobergrenze: die k besten Aktien zu gleichen Teilen
    k = int(np.ceil(1 / max_weight))
    hi = float(np.sort(mu)[::-1][:k].mean())
    vols, rets, weights = [], [], []
    for target in np.linspace(lo, hi, n_points):
        try:
            w = min_variance_for_target(mu, cov, target, max_weight)
        except RuntimeError:
            continue  # Zielrendite unter Gewichtsgrenzen nicht erreichbar
        weights.append(w)
        rets.append(portfolio_return(w, mu))
        vols.append(portfolio_vol(w, cov))
    return np.array(vols), np.array(rets), np.array(weights)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Relative Risikobeiträge RC_i = w_i * (Sigma w)_i / (w' Sigma w)."""
    marginal = cov @ w
    return w * marginal / (w @ marginal)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution (Spinu 2013).

    Konvexe Formulierung:  min 0.5 w'Σw - (1/n) Σ ln(w_i),  w > 0.
    Die Lösung wird auf Summe 1 normiert und erfüllt dann RC_i = 1/n.
    """
    n = len(cov)
    obj = lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n
    grad = lambda w: cov @ w - 1.0 / (n * w)
    x0 = 1.0 / np.sqrt(np.diag(cov))
    res = minimize(obj, x0 / x0.sum(), jac=grad, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    if not res.success:
        raise RuntimeError(f"Risk Parity fehlgeschlagen: {res.message}")
    return res.x / res.x.sum()
