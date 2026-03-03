import numpy as np
from riskmetrics.es import historical_es


def test_historical_es_returns_float():
    pnl = np.random.default_rng(1).normal(size=500)
    es = historical_es(pnl, alpha=0.95)
    assert isinstance(es, float)
