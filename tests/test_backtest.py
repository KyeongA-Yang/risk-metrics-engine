import numpy as np
import pandas as pd
from riskmetrics.backtest import var_violations, kupiec_pof_test


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


