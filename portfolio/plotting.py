"""Visualisierungen."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import drawdown_series

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frontier(vols, rets, stock_vols, stock_rets, labels, special: dict, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(vols, rets, lw=2, label="Effizienzkurve (long-only)")
    ax.scatter(stock_vols, stock_rets, c="gray", s=30)
    for x, y, l in zip(stock_vols, stock_rets, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    for name, (v, r) in special.items():
        ax.scatter(v, r, s=90, marker="*", label=name, zorder=5)
    ax.set(xlabel="Volatilität p.a.", ylabel="Erwartete Rendite p.a.", title="Mean-Variance-Effizienzkurve (In-Sample)")
    ax.legend()
    _save(fig, path)


def plot_correlation(corr: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    fig.colorbar(im)
    ax.set_title("Korrelationsmatrix")
    _save(fig, path)


def plot_risk_contributions(w_dict: dict, rc_dict: dict, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in zip(axes, (w_dict, rc_dict), ("Kapitalgewichte", "Risikobeiträge")):
        pd.DataFrame(data, index=labels).plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Anteil")
    _save(fig, path)


def plot_equity_curves(rets: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    (1 + rets).cumprod().plot(ax=ax, lw=1.6)
    ax.set(title="Backtest: Kumulierte Wertentwicklung (netto nach Kosten)", ylabel="Vermögen (Start = 1)")
    _save(fig, path)


def plot_drawdowns(rets: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in rets:
        drawdown_series(rets[c]).plot(ax=ax, lw=1.2, label=c)
    ax.set(title="Drawdowns", ylabel="Drawdown")
    ax.legend()
    _save(fig, path)


def plot_weights(weights: pd.DataFrame, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns)
    ax.set(title=title, ylim=(0, 1), ylabel="Gewicht")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    _save(fig, path)
