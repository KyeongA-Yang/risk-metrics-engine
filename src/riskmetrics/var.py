from __future__ import annotations

import numpy as np
import pandas as pd


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


def rolling_historical_var(pnl: "pd.Series", window: int, alpha: float = 0.99) -> "pd.Series":
    """
    Rolling historical VaR (positive loss threshold) computed from PnL series.

    Convention:
      - pnl > 0 profit, pnl < 0 loss
      - VaR returned as positive number (loss magnitude)

    Method:
      loss = -pnl
      VaR_t = quantile(loss_{t-window+1:t}, alpha)
           = - quantile(pnl_{t-window+1:t}, 1-alpha)    (equivalent)

    Returns a Series with the same index as pnl.
    First window-1 entries are NaN.
    """
    if not isinstance(pnl, pd.Series):
        raise TypeError("pnl must be a pandas Series")
    if window < 2:
        raise ValueError("window must be >= 2")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    
    # Ensure numeric
    pnl_numeric = pd.to_numeric(pnl, errors="coerce")
    # If non-numeric produced NaN, that's fine; rolling quantile will propagate NaNs
    # but we can be stricter if desired:
    # if pnl_numeric.isna().any(): raise ValueError("pnl contains non-numeric values")

    loss = -pnl_numeric

    out = loss.rolling(window=window, min_periods=window).quantile(alpha)

    # Ensure float dtype
    return out.astype(float)