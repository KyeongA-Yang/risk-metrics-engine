from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import var_violations, backtest_report


def load_pnl_from_csv(path: str, mode: str, col: str = "pnl") -> pd.Series:
    """
    mode:
      - "pnl": CSV contains pnl column (default: pnl)
      - "price": CSV contains date + price columns -> pnl = log(price).diff()
    Returns: pnl as pd.Series indexed by date (DatetimeIndex)
    """
    df = pd.read_csv(path)

    if mode == "pnl":
        # If CSV has no date column, synthesize one
        if "date" not in df.columns:
            df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in CSV: {path}")

        pnl = df[col].astype(float)
        return pnl

    if mode == "price":
        # Expect columns: date, price (SPY CSV)
        if "date" not in df.columns or "price" not in df.columns:
            raise SystemExit("For mode=price, CSV must have columns: date, price")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        price = df["price"].astype(float)
        pnl = np.log(price).diff().dropna()
        return pnl

    raise SystemExit(f"Unknown mode: {mode}. Use 'pnl' or 'price'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest diagnostic plot: rolling VaR + violations + Kupiec POF report"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV")
    parser.add_argument(
        "--mode",
        choices=["pnl", "price"],
        default="pnl",
        help="pnl: CSV has pnl column (and optional date); price: CSV has date+price",
    )
    parser.add_argument("--col", default="pnl", help="PnL column name when mode=pnl")
    parser.add_argument("--alpha", type=float, default=0.99, help="VaR confidence level")
    parser.add_argument("--window", type=int, default=250, help="Rolling window size")
    parser.add_argument("--zoom-last", type=int, default=300, help="Plot only last N points")
    parser.add_argument(
        "--loss-only",
        action="store_true",
        help="Plot only positive losses (clip at 0) for readability",
    )
    parser.add_argument(
        "--out",
        default="data/backtest_plot.png",
        help="Output path for saved PNG plot",
    )
    args = parser.parse_args()

    pnl = load_pnl_from_csv(args.csv, mode=args.mode, col=args.col)
    loss = -pnl

    rvar = rolling_historical_var(pnl, window=args.window, alpha=args.alpha)

    # report + violations
    viol = var_violations(loss, rvar)
    rep = backtest_report(loss, rvar, alpha=args.alpha)

    print("=== backtest report ===")
    print(rep)

    # align for plotting
    aligned = pd.concat([loss.rename("loss"), rvar.rename("VaR")], axis=1).dropna()
    aligned["violation"] = (aligned["loss"] > aligned["VaR"]).astype(int)

    aligned_zoom = aligned.tail(args.zoom_last)
    v = aligned_zoom[aligned_zoom["violation"] == 1]

    # choose plotted loss series
    if args.loss_only:
        y = aligned_zoom["loss"].clip(lower=0.0)
        y_v = v["loss"].clip(lower=0.0)
        loss_label = "loss (positive only)"
    else:
        y = aligned_zoom["loss"]
        y_v = v["loss"]
        loss_label = "loss"

    plt.figure()
    plt.plot(aligned_zoom.index, y, label=loss_label)
    plt.plot(aligned_zoom.index, aligned_zoom["VaR"], label="rolling VaR")
    plt.scatter(v.index, y_v, marker="o", label="violation")

    title_prefix = "price→logret" if args.mode == "price" else "pnl"
    plt.title(f"Backtest ({title_prefix}): alpha={args.alpha}, window={args.window}")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()