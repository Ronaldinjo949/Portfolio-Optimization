"""Strategien: Funktion (Fenster historischer Renditen) -> Zielgewichte."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import estimators as est
from . import optimizers as opt

Strategy = Callable[[pd.DataFrame], np.ndarray]


def equal_weight_strategy() -> Strategy:
    return lambda window: opt.equal_weight(window.shape[1])


def min_variance_strategy(max_weight: float = 0.10) -> Strategy:
    def f(window):
        cov, _ = est.ledoit_wolf_cov(window)
        return opt.min_variance(cov, max_weight)
    return f


def max_sharpe_strategy(rf: float = 0.0, max_weight: float = 0.10, mean_shrink: float = 0.5) -> Strategy:
    def f(window):
        cov, _ = est.ledoit_wolf_cov(window)
        mu = est.mean_returns(window, shrink=mean_shrink)
        return opt.max_sharpe(mu, cov, rf, max_weight)
    return f


def risk_parity_strategy() -> Strategy:
    def f(window):
        cov, _ = est.ledoit_wolf_cov(window)
        return opt.risk_parity(cov)
    return f
