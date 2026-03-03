from __future__ import annotations

import numpy as np

from .var import historical_var


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
