import numpy as np
from riskmetrics.es import historical_es


def test_historical_es_returns_float():
    pnl = np.random.default_rng(1).normal(size=500)
    es = historical_es(pnl, alpha=0.95)
    assert isinstance(es, float)

def test_historical_es_matches_definition():
    import numpy as np
    from riskmetrics.var import historical_var
    from riskmetrics.es import historical_es

    pnl = np.array([0.01, -0.02, 0.0, -0.03, 0.02])
    alpha = 0.8

    loss = -pnl
    var_a = historical_var(pnl, alpha=alpha)

    expected = loss[loss >= var_a].mean()
    got = historical_es(pnl, alpha=alpha)

    assert np.isclose(got, expected)