import numpy as np
from riskmetrics.backtest import var_violations, kupiec_pof_test


def test_var_violations_shape():
    pnl = np.array([1.0, -2.0, 0.5, -0.1])
    v = 1.0
    viol = var_violations(pnl, v)
    assert viol.shape == pnl.shape


def test_kupiec_returns_keys():
    rng = np.random.default_rng(0)
    viol = rng.random(100) < 0.02
    out = kupiec_pof_test(viol, alpha=0.99)
    assert "lr_pof" in out and "x" in out and "n" in out
