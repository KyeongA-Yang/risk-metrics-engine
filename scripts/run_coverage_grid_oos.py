from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import backtest_report, backtest_report_oos


def load_spy_returns(csv_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Load SPY prices (date, price) CSV and return:
      pnl: log returns series (pnl_t)
      loss: -pnl_t
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    price = df["price"].astype(float)
    pnl = np.log(price).diff().dropna()
    loss = -pnl
    return pnl, loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage grid runner (SPY): same-day vs OOS")
    parser.add_argument("--csv", default="data/price_SPY.csv", help="Path to SPY price CSV (date, price)")
    parser.add_argument("--out", default="data/coverage_grid_spy_oos.csv", help="Output CSV for grid table")
    parser.add_argument("--alphas", default="0.975,0.99", help="Comma-separated alphas, e.g. 0.975,0.99")
    parser.add_argument("--windows", default="125,250,500", help="Comma-separated windows, e.g. 125,250,500")
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    windows = [int(x.strip()) for x in args.windows.split(",")]

    pnl, loss = load_spy_returns(args.csv)

    rows = []
    for alpha in alphas:
        for window in windows:
            rvar = rolling_historical_var(pnl, window=window, alpha=alpha)

            # same-day report (loss_t > VaR_t)
            rep_same = backtest_report(loss, rvar, alpha=alpha)
            rep_same.update({"alpha": alpha, "window": window, "mode": "same_day"})

            # out-of-sample report (loss_t > VaR_{t-1})
            rep_oos = backtest_report_oos(loss, rvar, alpha=alpha)
            rep_oos.update({"alpha": alpha, "window": window, "mode": "oos_shift1"})

            rows.append(rep_same)
            rows.append(rep_oos)

    out = pd.DataFrame(rows).sort_values(["alpha", "window", "mode"]).reset_index(drop=True)

    # nicer column order
    cols = ["alpha", "window", "mode", "n", "x", "expected_rate", "observed_rate", "lr_pof", "p_value"]
    out = out[cols]

    print("=== Coverage grid (SPY): same-day vs OOS ===")
    print(out.to_string(index=False))

    out.to_csv(args.out, index=False)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()