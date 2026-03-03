from __future__ import annotations

import argparse
import pandas as pd

from .var import historical_var, parametric_var_normal
from .es import historical_es


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk Metrics Engine (basic)")
    parser.add_argument("--csv", required=True, help="Path to CSV containing pnl column")
    parser.add_argument("--col", default="pnl", help="Column name for PnL")
    parser.add_argument("--alpha", type=float, default=0.99, help="Confidence level for VaR/ES")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not found in CSV")

    pnl = df[args.col].to_numpy()

    hvar = historical_var(pnl, alpha=args.alpha)
    pvar = parametric_var_normal(pnl, alpha=args.alpha)
    hes = historical_es(pnl, alpha=args.alpha)

    print(f"alpha={args.alpha}")
    print(f"Historical VaR: {hvar:.6f}")
    print(f"Parametric VaR (Normal): {pvar:.6f}")
    print(f"Historical ES: {hes:.6f}")


if __name__ == "__main__":
    main()
