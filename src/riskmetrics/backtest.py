from __future__ import annotations

import numpy as np


def var_violations(pnl: np.ndarray, var_value: float) -> np.ndarray:
    """
    Return boolean array indicating VaR violations.
    Violation occurs when Loss > VaR (i.e., -pnl > var_value).
    """
    x = np.asarray(pnl, dtype=float)
    if x.size == 0:
        raise ValueError("pnl must be non-empty")
    if var_value < 0:
        raise ValueError("var_value must be non-negative")

    loss = -x
    return loss > var_value


def kupiec_pof_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.
    Returns dict with counts and LR statistic.

    alpha is VaR confidence level (e.g., 0.99) => expected violation prob p = 1 - alpha
    """
    v = np.asarray(violations, dtype=bool)
    n = int(v.size)
    if n == 0:
        raise ValueError("violations must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    x = int(v.sum())
    p = 1.0 - alpha

    eps = 1e-12
    phat = max(min(x / n, 1 - eps), eps)

    import math

    ll0 = (n - x) * math.log(1 - p + eps) + x * math.log(p + eps)
    ll1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    lr = -2.0 * (ll0 - ll1)

    return {"n": n, "x": x, "p_expected": p, "p_hat": phat, "lr_pof": lr}
