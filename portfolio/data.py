"""Datenbeschaffung: Kurse via yfinance (mit CSV-Cache) und synthetische Daten für Offline-Tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Diversifiziertes Aktienuniversum für das Portfolio; SPY dient als Benchmark, nicht als Asset im Portfolio.
BENCHMARK_TICKER = "SPY"
DEFAULT_UNIVERSE = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "UNH": "UnitedHealth",
    "CVX": "Chevron",
    "HD": "Home Depot",
    "PG": "Procter & Gamble",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "INTC": "Intel",
    "AMD": "AMD",
    "IBM": "IBM",
    "C": "Citigroup",
    "WMT": "Walmart",
    "MCD": "McDonald's",
    "V": "Visa",
    "MA": "Mastercard",
    "CAT": "Caterpillar",
    "BA": "Boeing",
    "HON": "Honeywell",
    "NKE": "Nike",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "T": "AT&T",
    "KO": "Coca-Cola",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
}


def load_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    """Lädt bereinigte Schlusskurse; nutzt einen CSV-Cache, damit Ergebnisse reproduzierbar bleiben."""
    cache = Path(cache_dir) / f"prices_{'_'.join(sorted(tickers))}_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf  # Import hier, damit Tests ohne yfinance laufen

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].dropna(how="any")
    if prices.empty:
        raise RuntimeError("Keine Kursdaten erhalten – Ticker/Zeitraum/Internet prüfen.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Einfache Tagesrenditen."""
    return prices.pct_change().dropna(how="any")


def synthetic_prices(
    tickers: list[str], n_days: int = 2500, seed: int = 42, start: str = "2012-01-02"
) -> pd.DataFrame:
    """Simulierte Kurse mit Ein-Faktor-Struktur (nur für Offline-Demo und Tests)."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    beta = rng.uniform(0.2, 1.2, n)
    mu = rng.uniform(0.02, 0.10, n) / 252
    idio = rng.uniform(0.05, 0.25, n) / np.sqrt(252)
    market = rng.normal(0, 0.15 / np.sqrt(252), n_days)
    rets = mu + np.outer(market, beta) + rng.normal(0, 1, (n_days, n)) * idio
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
