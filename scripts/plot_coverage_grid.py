from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coverage grid results (observed vs expected)")
    parser.add_argument("--csv", default="data/coverage_grid_spy_oos.csv", help="Input grid CSV")
    parser.add_argument("--out", default="data/coverage_grid_spy_oos.png", help="Output png path")
    parser.add_argument("--mode", default="oos_shift1", choices=["same_day", "oos_shift1"], help="Which mode to plot")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # filter mode and sort
    d = df[df["mode"] == args.mode].copy()
    d = d.sort_values(["alpha", "window"])

    plt.figure()

    # one line per alpha
    for alpha, g in d.groupby("alpha"):
        g = g.sort_values("window")
        plt.plot(g["window"], g["observed_rate"], marker="o", label=f"observed (alpha={alpha})")
        # expected line at 1-alpha
        exp = float(g["expected_rate"].iloc[0])
        plt.plot(g["window"], [exp] * len(g), linestyle="--", marker=None, label=f"expected (1-alpha={exp:.3f})")

    plt.title(f"SPY coverage grid ({args.mode}): observed vs expected violation rate")
    plt.xlabel("window")
    plt.ylabel("violation rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)
    plt.show()


if __name__ == "__main__":
    main()