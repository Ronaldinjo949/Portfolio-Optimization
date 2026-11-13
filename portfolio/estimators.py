"""Schätzer für Erwartungswerte und Kovarianz (Schätzfehler ist der Kernpunkt der MVO-Kritik)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * TRADING_DAYS


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf-Shrinkage (2004) auf das skalierte Identitäts-Target  m*I.

    Sigma_shrunk = delta * m * I + (1 - delta) * S
    Gibt (annualisierte Kovarianz, delta) zurück.
    """
    X = returns.values - returns.values.mean(axis=0)
    T, N = X.shape
    S = X.T @ X / T
    m = np.trace(S) / N
    d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    # sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t||^4 - T * ||S||_F^2
    b2_bar = (np.sum(np.sum(X**2, axis=1) ** 2) - T * np.sum(S**2)) / (T**2 * N)
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0 else b2 / d2
    cov = delta * m * np.eye(N) + (1 - delta) * S
    return cov * TRADING_DAYS, float(delta)


def mean_returns(returns: pd.DataFrame, shrink: float = 0.0) -> np.ndarray:
    """Annualisierte Erwartungsrenditen (arithmetisch).

    shrink in [0, 1]: Schrumpfung in Richtung des Querschnittsmittels
    (dämpft extreme, schätzfehlerbehaftete Renditeprognosen).
    """
    mu = returns.mean().values * TRADING_DAYS
    return (1 - shrink) * mu + shrink * mu.mean()
