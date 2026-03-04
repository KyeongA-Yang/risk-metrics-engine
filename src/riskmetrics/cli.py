from __future__ import annotations

import argparse
import pandas as pd

from .var import historical_var, parametric_var_normal, rolling_historical_var
from .es import historical_es


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk Metrics Engine (basic)")
    parser.add_argument("--csv", required=True, help="Path to CSV containing pnl column")
    parser.add_argument("--col", default="pnl", help="Column name for PnL")
    parser.add_argument("--alpha", type=float, default=0.99, help="Confidence level for VaR/ES")
    parser.add_argument("--rolling-window", type=int, default=None, help="Rolling window size for historical VaR")
    args = parser.parse_args()

   # (1) CSV 읽기
    df = pd.read_csv(args.csv)

    # (2) 컬럼 존재 확인 (없으면 종료)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not found in CSV")

    # (3) numpy 버전 (기존 함수들이 numpy 입력 받는 형태라 유지)
    pnl = df[args.col].to_numpy()

    # (4) pandas Series 버전 (rolling 함수는 Series가 필요)
    pnl_series = df[args.col]

    # (5) 단일(전체) VaR/ES 계산
    hvar = historical_var(pnl, alpha=args.alpha)
    pvar = parametric_var_normal(pnl, alpha=args.alpha)
    hes = historical_es(pnl, alpha=args.alpha)

    # (6) rolling-window 옵션이 있으면 rolling VaR 계산/출력
    if args.rolling_window is not None:
        rvar = rolling_historical_var(pnl_series, window=args.rolling_window, alpha=args.alpha)
        tail = rvar.dropna().tail(5)

        print("\nRolling Historical VaR (last 5):")
        for idx, val in tail.items():
            print(f"{idx}: {val:.6f}")

    # (7) 요약 출력
    print(f"\nalpha={args.alpha}")
    print(f"Historical VaR: {hvar:.6f}")
    print(f"Parametric VaR (Normal): {pvar:.6f}")
    print(f"Historical ES: {hes:.6f}")


if __name__ == "__main__":
    main()
