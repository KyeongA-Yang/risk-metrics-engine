from __future__ import annotations

import numpy as np
import pandas as pd

from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import backtest_report


def load_spy_loss(path: str = "data/price_SPY.csv") -> pd.Series:
    """
    Load SPY prices and return a loss Series indexed by date.
    loss_t = - log_return_t
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    price = df["price"].astype(float)
    ret = np.log(price).diff()  # log return
    loss = -ret                 # loss = -return
    loss = loss.dropna()
    return loss


def main() -> None:
    alphas = [0.975, 0.99]
    windows = [125, 250, 500]

    loss = load_spy_loss("data/price_SPY.csv")

    rows: list[dict] = []

    for alpha in alphas:
        for window in windows:
            # rolling_historical_var expects pnl/returns series (profit positive)
            pnl = -loss

            rvar = rolling_historical_var(pnl, window=window, alpha=alpha)

            rep = backtest_report(loss, rvar, alpha=alpha)

            rows.append(
                {
                    "alpha": alpha,
                    "window": window,
                    "n": rep["n"],
                    "x": rep["x"],
                    "expected_rate": rep["expected_rate"],
                    "observed_rate": rep["observed_rate"],
                    "lr_pof": rep["lr_pof"],
                    "p_value": rep["p_value"],
                }
            )

    out = pd.DataFrame(rows).sort_values(["alpha", "window"]).reset_index(drop=True)

    # Pretty printing
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")

    print("=== Coverage grid (SPY) ===")
    print(out)

    # Optional: save table for reporting
    out_path = "data/coverage_grid_spy.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()