from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import var_violations

alpha = 0.99
window = 250

df = pd.read_csv("data/long_pnl.csv")
df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
df = df.set_index("date").sort_index()

pnl = df["pnl"]
loss = -pnl

rvar = rolling_historical_var(pnl, window=window, alpha=alpha)
viol = var_violations(loss, rvar)

aligned = pd.concat([loss.rename("loss"), rvar.rename("VaR")], axis=1).dropna()
aligned["violation"] = (aligned["loss"] > aligned["VaR"]).astype(int)

plt.figure()
plt.plot(aligned.index, aligned["loss"], label="loss")
plt.plot(aligned.index, aligned["VaR"], label="rolling VaR")

v = aligned[aligned["violation"] == 1]
plt.scatter(v.index, v["loss"], marker="o", label="violation")

plt.title(f"Loss vs rolling VaR (alpha={alpha}, window={window})")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.tight_layout()

plt.savefig("data/backtest_plot.png", dpi=200)
plt.show()