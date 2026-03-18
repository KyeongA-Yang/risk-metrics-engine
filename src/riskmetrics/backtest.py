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


def var_violations_oos(loss: pd.Series, var_t: pd.Series) -> pd.Series:
    """
    Out-of-sample violations:
    compare realized loss_t vs VaR_{t-1} (1-step shift).

    Inputs:
      - loss: realized loss series indexed by date t
      - var_t: VaR series computed using data up to time t (rolling, includes t)

    Output:
      - violations indexed where both loss_t and var_{t-1} are available
    """
    if not isinstance(loss, pd.Series) or not isinstance(var_t, pd.Series):
        raise TypeError("loss and var_t must be pandas Series")

    var_prev = var_t.shift(1)  # VaR_{t-1} aligned to time t
    aligned = pd.concat([loss.rename("loss"), var_prev.rename("var_prev")], axis=1).dropna()
    return (aligned["loss"] > aligned["var_prev"]).astype(int)


def backtest_report_oos(loss: pd.Series, var_t: pd.Series, alpha: float) -> dict:
    """
    Compact out-of-sample backtest report:
    - violations defined as loss_t > VaR_{t-1}
    - Kupiec POF test
    """
    viol = var_violations_oos(loss, var_t)
    out = kupiec_pof_test(viol, alpha=alpha)
    return {
        "n": out["n"],
        "x": out["x"],
        "expected_rate": out["expected_rate"],
        "observed_rate": out["observed_rate"],
        "lr_pof": out["lr_pof"],
        "p_value": out["p_value"],
    }


def es_backtest_report(loss: pd.Series, var: pd.Series, es:pd.Series, alpha: float) -> dict:
    """
    ES backtest-style summary (severity-focused), using same-day VaR threshold.

    - Align (loss, var, es) by index
    - Define violations by VaR: loss_t > VaR_t
    - Summarize tail severity on violation days:
        * mean_loss_given_violation
        * mean_es_given_violation
        * mean_excess_over_var = E[loss - VaR | violation]
        * mean_excess_over_es  = E[loss - ES  | violation]

    Notes:
    - ES is not typically "backtested" by a simple coverage count like VaR.
      Here we report severity diagnostics conditional on VaR violations.
    """
    if not isinstance(loss, pd.Series) or not isinstance(var, pd.Series) or not isinstance(es, pd.Series):
        raise TypeError("loss, var, es must be pandas Series")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    aligned = pd.concat(
        [loss.rename("loss"), var.rename("var"), es.rename("es")],
        axis=1,
    ).dropna()

    n = int(len(aligned))
    if n == 0:
        raise ValueError("Aligned series is empty (check indices / NaNs).")

    viol = (aligned["loss"] > aligned["var"]).astype(int)
    x = int(viol.sum())

    expected_rate = 1.0 - alpha
    observed_rate = float(x / n)

    if x == 0:
        # No violation days → severity stats undefined
        return {
            "n": n,
            "x": x,
            "expected_rate": expected_rate,
            "observed_rate": observed_rate,
            "mean_loss_given_violation": float("nan"),
            "mean_es_given_violation": float("nan"),
            "mean_excess_over_var": float("nan"),
            "mean_excess_over_es": float("nan"),
        }

    tail = aligned.loc[viol == 1]
    mean_loss = float(tail["loss"].mean())
    mean_es = float(tail["es"].mean())
    mean_excess_var = float((tail["loss"] - tail["var"]).mean())
    mean_excess_es = float((tail["loss"] - tail["es"]).mean())

    return {
        "n": n,
        "x": x,
        "expected_rate": expected_rate,
        "observed_rate": observed_rate,
        "mean_loss_given_violation": mean_loss,
        "mean_es_given_violation": mean_es,
        "mean_excess_over_var": mean_excess_var,
        "mean_excess_over_es": mean_excess_es,
    }


def es_backtest_report_oos(loss: pd.Series, var_t: pd.Series, es_t: pd.Series, alpha: float) -> dict:
    """
    Out-of-sample (1-step shift) ES severity report:

    - Compare realized loss_t vs VaR_{t-1}
    - Use ES_{t-1} as the model ES available at time t
    - Summarize tail severity conditional on VaR_{t-1} violations:
        loss_t > VaR_{t-1}

    This is more deployment-faithful (no look-ahead).
    """
    if not isinstance(loss, pd.Series) or not isinstance(var_t, pd.Series) or not isinstance(es_t, pd.Series):
        raise TypeError("loss, var_t, es_t must be pandas Series")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    var_prev = var_t.shift(1).rename("var_prev")
    es_prev = es_t.shift(1).rename("es_prev")

    aligned = pd.concat(
        [loss.rename("loss"), var_prev, es_prev],
        axis=1,
    ).dropna()

    n = int(len(aligned))
    if n == 0:
        raise ValueError("Aligned series is empty (check indices / NaNs).")

    viol = (aligned["loss"] > aligned["var_prev"]).astype(int)
    x = int(viol.sum())

    expected_rate = 1.0 - alpha
    observed_rate = float(x / n)

    if x == 0:
        return {
            "n": n,
            "x": x,
            "expected_rate": expected_rate,
            "observed_rate": observed_rate,
            "mean_loss_given_violation": float("nan"),
            "mean_es_given_violation": float("nan"),
            "mean_excess_over_var": float("nan"),
            "mean_excess_over_es": float("nan"),
        }

    tail = aligned.loc[viol == 1]
    mean_loss = float(tail["loss"].mean())
    mean_es = float(tail["es_prev"].mean())
    mean_excess_var = float((tail["loss"] - tail["var_prev"]).mean())
    mean_excess_es = float((tail["loss"] - tail["es_prev"]).mean())

    return {
        "n": n,
        "x": x,
        "expected_rate": expected_rate,
        "observed_rate": observed_rate,
        "mean_loss_given_violation": mean_loss,
        "mean_es_given_violation": mean_es,
        "mean_excess_over_var": mean_excess_var,
        "mean_excess_over_es": mean_excess_es,
    }

