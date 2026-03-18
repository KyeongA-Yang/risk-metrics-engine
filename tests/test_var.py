import numpy as np
import pandas as pd

from riskmetrics.var import historical_var, parametric_var_normal, rolling_historical_var


def test_historical_var_returns_float():
    pnl = np.array([1.0, -1.0, 2.0, -2.0])
    v = historical_var(pnl, alpha=0.5)
    assert isinstance(v, float)


def test_parametric_var_normal_not_nan():
    pnl = np.random.default_rng(0).normal(size=200)
    v = parametric_var_normal(pnl, alpha=0.99)
    assert not np.isnan(v)


def test_historical_var_matches_numpy_quantile():
    pnl = np.array([0.01, -0.02, 0.0, -0.03, 0.02])
    alpha = 0.8

    loss = -pnl
    expected = np.quantile(loss, alpha, method="linear")
    got = historical_var(pnl, alpha=alpha)

    assert np.isclose(got, expected)


def test_rolling_historical_var_preserves_index_and_nans():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    pnl = pd.Series([0.01, -0.02, 0.00, -0.03, 0.02, -0.01], index=idx)

    window = 3
    alpha = 0.99

    rvar = rolling_historical_var(pnl, window=window, alpha=alpha)

    assert len(rvar) == len(pnl)
    assert rvar.index.equals(pnl.index)
    assert rvar.iloc[: window - 1].isna().all()


def test_rolling_historical_var_last_value_matches_manual_quantile():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    pnl = pd.Series([0.01, -0.02, 0.00, -0.03, 0.02, -0.01], index=idx)

    window = 3
    alpha = 0.99

    rvar = rolling_historical_var(pnl, window=window, alpha=alpha)

    pnl_window = pnl.iloc[-window:]
    loss_window = -pnl_window.to_numpy()
    expected = np.quantile(loss_window, alpha, method="linear")

    got = float(rvar.iloc[-1])
    assert np.isclose(got, expected)