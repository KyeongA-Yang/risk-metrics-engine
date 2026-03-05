from __future__ import annotations

import argparse
import pandas as pd

from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import var_violations, kupiec_pof_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest demo: rolling VaR + Kupiec POF")
    parser.add_argument("--csv", required=True, help="path to CSV containing pnl column")
    parser.add_argument("--col", default="pnl", help="Column name for PnL")
    parser.add_argument("--alpha", type=float, default=0.99, help="VaR/ES confidence level")
    parser.add_argument("--window", type=int, default=250, help="Rolling window size")
    parser.add_argument(
        "--add-date",
        action="store_true",
        help="Add a synthetic date index (useful if CSV has no date column)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        raise SystemError(f"Column '{args.col}' not found in CSV")

    if args.add_date:
        df["date"] = pd.date_range("2026-01-01", periods=len(df), freq="D")
        df = df.set_index("date").sort_index()

    pnl = df[args.col]
    loss = -pnl

    rvar = rolling_historical_var(pnl, window=args.window, alpha=args.alpha)
    viol = var_violations(loss, rvar)

    out = kupiec_pof_test(viol, alpha=args.alpha)

    print("=== Backtest summary ===")
    print(f"alpha={args.alpha}  window={args.window}")
    print(f"n={out['n']}  x={out['x']}")
    print(f"expected_rate={out['expected_rate']:.6f}")
    print(f"observed_rate={out['observed_rate']:.6f}")
    print(f"LR_POF={out['lr_pof']:.6f}")
    print(f"p_value={out['p_value']:.6g}")

    print("\n=== Last 5 rolling VaR ===")
    print(rvar.dropna().tail(5).to_string())

    print("\n=== Last 5 violations ===")
    print(viol.tail(5).to_string())


if __name__ == "__main__":
    main()