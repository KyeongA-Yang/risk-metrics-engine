from __future__ import annotations

import numpy as np


def historical_var(pnl: np.ndarray, alpha: float = 0.99) -> float:
    """
    Historical (empirical) Value-at-Risk for *losses*.

    Convention:
      - pnl > 0 means profit, pnl < 0 means loss
      - VaR is returned as a positive number representing loss threshold

    VaR_alpha = quantile of loss at alpha (e.g., 99%).
    """
    x = np.asarray(pnl, dtype=float)
    if x.size == 0:
        raise ValueError("pnl must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    loss = -x
    q = np.quantile(loss, alpha, method="linear")
    return float(q)


def parametric_var_normal(pnl: np.ndarray, alpha: float = 0.99, ddof: int = 1) -> float:
    """
    Parametric VaR under Normal assumption on losses.

    VaR_alpha = mu_loss + z_alpha * sigma_loss
    """
    from scipy.stats import norm

    x = np.asarray(pnl, dtype=float)
    if x.size == 0:
        raise ValueError("pnl must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    loss = -x
    mu = float(np.mean(loss))
    sigma = float(np.std(loss, ddof=ddof))
    z = float(norm.ppf(alpha))
    return mu + z * sigma
