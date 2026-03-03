import numpy as np
from riskmetrics.var import historical_var, parametric_var_normal


def test_historical_var_returns_float():
    pnl = np.array([1.0, -1.0, 2.0, -2.0])
    v = historical_var(pnl, alpha=0.5)
    assert isinstance(v, float)


def test_parametric_var_normal_not_nan():
    pnl = np.random.default_rng(0).normal(size=200)
    v = parametric_var_normal(pnl, alpha=0.99)
    assert not np.isnan(v)

def test_historical_var_matches_numpy_quantile():
    import numpy as np
    from riskmetrics.var import historical_var

    pnl = np.array([0.01, -0.02, 0.0, -0.03, 0.02])
    alpha = 0.8

    loss = -pnl
    expected = np.quantile(loss, alpha, method="linear")
    got = historical_var(pnl, alpha=alpha)

    assert np.isclose(got, expected)