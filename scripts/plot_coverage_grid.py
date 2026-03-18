from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_one_mode(df: pd.DataFrame, mode: str, ax=None) -> None:
    d = df[df["mode"] == mode].copy().sort_values(["alpha", "window"])
    if ax is None:
        ax = plt.gca()

    for alpha, g in d.groupby("alpha"):
        g = g.sort_values("window")
        ax.plot(g["window"], g["observed_rate"], marker="o", label=f"observed (alpha={alpha:g})")
        exp = float(g["expected_rate"].iloc[0])
        ax.plot(g["window"], [exp] * len(g), linestyle="--", label=f"expected (1-alpha={exp:.3f})")

    ax.set_title(mode)
    ax.set_xlabel("window")
    ax.set_ylabel("violation rate")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coverage grid results (observed vs expected)")
    parser.add_argument("--csv", default="data/coverage_grid_spy_oos.csv", help="Input grid CSV")
    parser.add_argument("--out", default="data/coverage_grid_spy_oos.png", help="Output png path")
    parser.add_argument(
        "--mode",
        default="oos_shift1",
        choices=["same_day", "oos_shift1", "both"],
        help="Which mode to plot",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure numeric types (safety)
    df["alpha"] = df["alpha"].astype(float)
    df["window"] = df["window"].astype(int)
    df["expected_rate"] = df["expected_rate"].astype(float)
    df["observed_rate"] = df["observed_rate"].astype(float)

    if args.mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        plot_one_mode(df, "same_day", ax=axes[0])
        plot_one_mode(df, "oos_shift1", ax=axes[1])
        axes[0].legend(fontsize=8)
        fig.suptitle("SPY coverage grid: observed vs expected violation rate")
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        print("Saved:", args.out)
        plt.show()
        return

    # single mode
    plt.figure()
    plot_one_mode(df, args.mode)
    plt.title(f"SPY coverage grid ({args.mode}): observed vs expected violation rate")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)
    plt.show()


if __name__ == "__main__":
    main()