import numpy as np
import pandas as pd

from riskmetrics.backtest import var_violations, kupiec_pof_test, es_backtest_report, es_backtest_report_oos


def test_var_violations_basic():
    loss = pd.Series([0.01, 0.04, 0.03])
    var = pd.Series([0.03, 0.03, 0.03])
    viol = var_violations(loss, var)
    assert viol.tolist() == [0, 1, 0]


def test_kupiec_returns_keys():
    rng = np.random.default_rng(0)
    viol = pd.Series(rng.random(100) < 0.02)  # Series로 감싸기
    out = kupiec_pof_test(viol, alpha=0.99)
    assert {"lr_pof", "x", "n", "p_value"} <= set(out.keys())


def test_es_backtest_report_keys():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    loss = pd.Series([0.01, 0.02, 0.03, 0.04, 0.01, 0.05], index=idx)
    var = pd.Series([0.03] * 6, index=idx)
    es = pd.Series([0.04] * 6, index=idx)

    out = es_backtest_report(loss, var, es, alpha=0.99)
    expected_keys = {
        "n", "x", "expected_rate", "observed_rate",
        "mean_loss_given_violation", "mean_es_given_violation",
        "mean_excess_over_var", "mean_excess_over_es",
    }
    assert expected_keys <= set(out.keys())


def test_es_backtest_report_nan_when_no_violations():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    loss = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    var = pd.Series([0.10] * 5, index=idx)  # never violated
    es = pd.Series([0.12] * 5, index=idx)

    out = es_backtest_report(loss, var, es, alpha=0.99)
    assert out["x"] == 0
    assert np.isnan(out["mean_loss_given_violation"])


def test_es_backtest_report_oos_shifts_one_step():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    loss = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    var_t = pd.Series([0.03] * 5, index=idx)
    es_t = pd.Series([0.04] * 5, index=idx)

    out = es_backtest_report_oos(loss, var_t, es_t, alpha=0.99)

    # Because we shift var/es by 1, effective n must be <= len(idx)-1 (plus any NaNs)
    assert out["n"] <= 4
    # keys exist
    assert "mean_excess_over_es" in out