# Portfolio-Optimierung: Mean-Variance, Risk Parity & Walk-Forward-Backtest

Vergleich von vier Allokationsverfahren auf einem Aktienuniversum mit 30 Einzelaktien
mit **realistischem Out-of-Sample-Backtest** (kein Look-ahead, Transaktionskosten, Gewichtsdrift).

## Methodik

| Baustein | Umsetzung |
|---|---|
| Mean-Variance-Optimierung | Min-Varianz, Max-Sharpe, Effizienzkurve (SLSQP, long-only, Gewichtsobergrenze) |
| Risk Parity (ERC) | Konvexe Formulierung nach Spinu (2013): min ½w'Σw − (1/n)Σ ln wᵢ |
| Kovarianzschätzung | Ledoit-Wolf-Shrinkage (2004), selbst implementiert |
| Renditeschätzung | Historischer Mittelwert mit Shrinkage zum Querschnittsmittel |
| Backtest | Rollierendes Schätzfenster (252 Tage), monatliches Rebalancing, 10 bp Kosten |
| Kennzahlen | CAGR, Volatilität, Sharpe, Sortino, Max Drawdown, Calmar, VaR/CVaR (95 %) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py              # echte Daten (yfinance)
python run_analysis.py --synthetic  # Offline-Demo
pytest                              # Tests
```

Ergebnisse (Grafiken + `backtest_summary.csv`) landen in `results/`.

## Projektstruktur

```
portfolio/
  data.py         Datenabruf + Cache, synthetische Daten
  estimators.py   Ledoit-Wolf, Mittelwert-Shrinkage
  optimizers.py   Min-Var, Max-Sharpe, Frontier, Risk Parity
  strategies.py   Strategien: Renditefenster -> Gewichte
  backtest.py     Walk-forward-Engine
  metrics.py      Kennzahlen
  plotting.py     Grafiken
tests/            Unit-Tests (u. a. Look-ahead-Test, ERC-Eigenschaft)
run_analysis.py   Einstiegspunkt
```

## Ergebnisse

_Hier nach dem Lauf mit echten Daten die Grafiken aus `results/` einbinden und 3–5 Kernaussagen formulieren:_

<!-- ![Equity Curves](results/equity_curves.png) -->

## Wichtige Erkenntnisse / Limitationen

- Mean-Variance reagiert stark auf Schätzfehler in den Erwartungsrenditen (Michaud: "error maximizer") → Shrinkage und Gewichtsgrenzen.
- Risk Parity hängt nur von der Kovarianz ab und ist dadurch stabiler (geringerer Turnover); bei Einzelaktien ist die Sensitivität gegenüber Sektorkorrelationen höher.
- Effizienzkurve ist In-Sample; aussagekräftig ist nur der Out-of-Sample-Backtest.
- Keine Steuern, kein Slippage-Modell, Survivorship-Bias bei der Aktienauswahl möglich.

## Literatur

Markowitz (1952); Ledoit & Wolf (2004); Maillard, Roncalli & Teïletche (2010); Spinu (2013); Michaud (1989).
