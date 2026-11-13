import numpy as np

from portfolio import estimators as est
from portfolio import optimizers as opt
from portfolio.backtest import run_backtest
from portfolio.data import DEFAULT_UNIVERSE, synthetic_prices, to_returns
from portfolio.strategies import equal_weight_strategy, risk_parity_strategy

TICKERS = list("ABCDE")


def test_default_universe_is_stock_basket_and_not_spy():
    assert "SPY" not in DEFAULT_UNIVERSE
    assert {"AAPL", "MSFT", "NVDA", "AMZN", "META"}.issubset(set(DEFAULT_UNIVERSE))


def _rets():
    return to_returns(synthetic_prices(TICKERS, n_days=800))


def test_weights_sum_to_one_and_long_only():
    r = _rets()
    cov, _ = est.ledoit_wolf_cov(r)
    mu = est.mean_returns(r)
    for w in (opt.min_variance(cov), opt.max_sharpe(mu, cov), opt.risk_parity(cov)):
        assert abs(w.sum() - 1) < 1e-8
        assert (w >= -1e-9).all()


def test_min_variance_beats_equal_weight():
    cov, _ = est.ledoit_wolf_cov(_rets())
    assert opt.portfolio_vol(opt.min_variance(cov), cov) <= opt.portfolio_vol(opt.equal_weight(5), cov) + 1e-10


def test_risk_parity_equal_risk_contributions():
    cov, _ = est.ledoit_wolf_cov(_rets())
    rc = opt.risk_contributions(opt.risk_parity(cov), cov)
    assert np.allclose(rc, 1 / 5, atol=1e-4)


def test_ledoit_wolf_shrinkage_valid_and_psd():
    cov, delta = est.ledoit_wolf_cov(_rets())
    assert 0 <= delta <= 1
    assert np.linalg.eigvalsh(cov).min() > 0


def test_frontier_monotone():
    r = _rets()
    cov, _ = est.ledoit_wolf_cov(r)
    vols, rets, _ = opt.efficient_frontier(est.mean_returns(r), cov, n_points=15)
    assert np.all(np.diff(vols) >= -1e-6) and np.all(np.diff(rets) >= 0)


def test_backtest_no_lookahead():
    """Ändert man die Zukunft, dürfen frühere Backtest-Renditen sich nicht ändern."""
    r = _rets()
    r2 = r.copy()
    r2.iloc[-50:] *= 3
    a = run_backtest(r, risk_parity_strategy(), lookback=120).returns
    b = run_backtest(r2, risk_parity_strategy(), lookback=120).returns
    cut = r.index[-51]
    assert np.allclose(a[:cut], b[:cut])


def test_costs_reduce_returns():
    r = _rets()
    free = run_backtest(r, equal_weight_strategy(), lookback=120, cost_bps=0).returns
    paid = run_backtest(r, equal_weight_strategy(), lookback=120, cost_bps=50).returns
    assert paid.sum() < free.sum()
