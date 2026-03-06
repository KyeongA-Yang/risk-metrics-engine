from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def var_violations(loss: pd.Series, var: pd.Series) -> pd.Series:
    """
    Return VaR violation indicator series (1 if loss > VaR, else 0).

    Notes:
    - Uses pandas index alignment via concat + dropna.
    - Output index is the aligned index where both loss and var are present.  
    """
    if not isinstance(loss, pd.Series) or not isinstance(var, pd.Series):
        raise TypeError("loss and var must be pandas Series")
    
    aligned = pd.concat([loss.rename("loss"), var.rename("var")], axis=1).dropna()

    # violation occurs when realized loss exceeds VaR threshold
    viol = (aligned["loss"] > aligned["var"]).astype(int)
    return viol


def kupiec_pof_test(violations: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) likelihood ratio test + p-value.

    H0: violation probability p0 = 1 - alpha
    H1: p estimated by phat = x / n

    Returns:
      dict with n, x, expected_rate, observed_rate, lr_pof, p-value
    """
    if not isinstance(violations, pd.Series):
        violations = pd.Series(violations)

    v = violations.dropna().astype(int).to_numpy()
    n = int(v.size)
    if n == 0:
        raise ValueError("violations must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    x = int(v.sum())
    p0 = 1.0 - alpha

    # avoid log(0) when x=0 or x=n
    eps = 1e-12
    p0c = min(max(p0, eps), 1 - eps)
    phat = x / n
    phatc = min(max(phat, eps), 1 - eps)

    #binomial log-likelihood (up to constant)
    ll0 = x * np.log(p0c) + (n - x) * np.log(1 - p0c)
    ll1 = x * np.log(phatc) + (n - x) * np.log(1 - phatc)

    lr = float(-2.0 * (ll0 - ll1))
    p_value = float(1.0 - chi2.cdf(lr, df=1))

    return {
        "n": n,
        "x": x,
        "expected_rate": p0,
        "observed_rate": phat,
        "lr_pof": lr,
        "p_value": p_value
    }


def backtest_report(loss: pd.Series, var: pd.Series, alpha: float) -> dict:
    """
    Compact backtest report for VaR coverage:
    - align loss and var
    - violations
    - Kupiec POF test (LR + p-value)
    """
    viol = var_violations(loss, var)
    out = kupiec_pof_test(viol, alpha=alpha)

    # 더 보기 좋게 요약
    return {
        "n": out["n"],
        "x": out["x"],
        "expected_rate": out["expected_rate"],
        "observed_rate": out["observed_rate"],
        "lr_pof": out["lr_pof"],
        "p_value": out["p_value"],
    }

