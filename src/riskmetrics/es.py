from __future__ import annotations

import numpy as np
import pandas as pd

from .var import historical_var, rolling_historical_var


def historical_es(pnl: np.ndarray, alpha: float = 0.99) -> float:
    """
    Historical Expected Shortfall (ES) for losses.

    ES_alpha = E[Loss | Loss >= VaR_alpha]
    Returned as positive number (loss).
    """
    x = np.asarray(pnl, dtype=float)
    if x.size == 0:
        raise ValueError("pnl must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    loss = -x
    var_a = historical_var(pnl, alpha=alpha)
    tail = loss[loss >= var_a]

    if tail.size == 0:
        tail = loss[loss >= np.max(loss)]

    return float(np.mean(tail))


def rolling_historical_es(pnl: pd.Series, window: int, alpha: float = 0.99) -> pd.Series:
    """
    Rolling historical ES (Expected Shortfall) computed from a PnL series.

    loss = -pnl
    VaR_t = quantile(loss_window, alpha)
    ES_t  = mean(loss_window[loss_window >= VaR_t])
    """
    if not isinstance(pnl, pd.Series):
        raise TypeError("pnl must be a pandas Series")
    if window < 2:
        raise ValueError("window must be >= 2")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    pnl_numeric = pd.to_numeric(pnl, errors="coerce")
    loss = (-pnl_numeric).astype(float)

    # compute rolling ES directly from loss windows
    def es_win(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        v = float(np.quantile(arr, alpha, method="linear"))
        tail = arr[arr >= v]
        if tail.size == 0:
            return np.nan
        return float(tail.mean())

    out = loss.rolling(window=window, min_periods=window).apply(
        lambda s: es_win(s.to_numpy()), raw=False
    )
    return out.astype(float)