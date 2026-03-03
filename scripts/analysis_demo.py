import pandas as pd
import numpy as np

# (1) 데이터 읽기
df = pd.read_csv("data/sample_pnl.csv")
pnl = df["pnl"].to_numpy()
loss = -pnl

# (2) 요약통계
print("=== pnl summary ===")
print(pd.Series(pnl).describe())

print("\n=== loss summary ===")
print(pd.Series(pnl).describe())

# (3) VaR/ES 직접 계산
alpha = 0.99
var_hist = np.quantile(loss, alpha, method="linear")
es_hist = loss[loss >= var_hist].mean()

print(f"\nalpha={alpha}")
print(f"VaR (direct, historical): {var_hist:.6f}")
print(f"ES  (direct, historical): {es_hist:.6f}")

# (4) 우리가 만든 함수로 계산 (검증)

from riskmetrics.var import historical_var
from riskmetrics.es import historical_es

print("\n=== using our functions ===")
print(f"Historical VaR: {historical_var(pnl, alpha=alpha):.6f}")
print(f"Historical ES : {historical_es(pnl, alpha=alpha):.6f}")
