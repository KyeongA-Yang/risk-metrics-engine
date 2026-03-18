import numpy as np
import pandas as pd

from riskmetrics.es import historical_es, rolling_historical_es
from riskmetrics.var import historical_var, rolling_historical_var


def test_historical_es_returns_float():
    pnl = np.random.default_rng(1).normal(size=500)
    es = historical_es(pnl, alpha=0.95)
    assert isinstance(es, float)


def test_historical_es_matches_definition():
    pnl = np.array([0.01, -0.02, 0.0, -0.03, 0.02])
    alpha = 0.8

    loss = -pnl
    var_a = historical_var(pnl, alpha=alpha)
    expected = float(loss[loss >= var_a].mean())

    got = historical_es(pnl, alpha=alpha)
    assert np.isclose(got, expected)


def test_rolling_es_preserves_index_and_nans():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    pnl = pd.Series([0.01, -0.02, 0.00, -0.03, 0.02, -0.01], index=idx)

    window = 3
    alpha = 0.99

    es = rolling_historical_es(pnl, window=window, alpha=alpha)

    assert len(es) == len(pnl)
    assert es.index.equals(pnl.index)
    assert es.iloc[: window - 1].isna().all()


def test_rolling_es_geq_rolling_var():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    pnl = pd.Series([0.015, -0.020, 0.005, -0.010, 0.012, -0.030, 0.008, -0.006], index=idx)

    window = 3
    alpha = 0.99

    rvar = rolling_historical_var(pnl, window=window, alpha=alpha)
    res = rolling_historical_es(pnl, window=window, alpha=alpha)

    aligned = pd.concat([rvar.rename("var"), res.rename("es")], axis=1).dropna()
    assert (aligned["es"] >= aligned["var"]).all()


def test_rolling_es_last_value_matches_manual_window():
    pnl = pd.Series([0.015, -0.020, 0.005, -0.010, 0.012, -0.030, 0.008, -0.006])
    window = 3
    alpha = 0.99

    es = rolling_historical_es(pnl, window=window, alpha=alpha)

    pnl_win = pnl.iloc[-window:]
    loss_win = (-pnl_win).to_numpy()

    var_win = float(np.quantile(loss_win, alpha, method="linear"))
    tail = loss_win[loss_win >= var_win]
    expected = float(tail.mean())

    got = float(es.dropna().iloc[-1])
    assert np.isclose(got, expected)