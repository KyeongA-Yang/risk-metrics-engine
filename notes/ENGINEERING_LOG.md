# Engineering Log
> Template: Context → Implementation → Pitfalls & Best Practices → How to run → Next

---

## 2026-03-03 — Project setup / tooling basics

### Context
- Set up a clean Python dev environment and initialize a reusable project structure for a risk-metrics package.
- 파이썬 프로젝트를 “정석 구조 + 재현 가능 실행”으로 시작하는 게 목표.
- 1순위 목표는 파이썬이랑 친해지면서 리스크에 대해서 공부하는 것!

### Implementation
- Installed `pyenv` and selected Python 3.12 (macOS).
- Created project venv `.venv` and installed dependencies in an isolated environment.
- Initialized repo structure:
  - `src/` (package), `tests/` (pytest), `scripts/` (runnable scripts), `notebooks/` (analysis), `data/` (sample input)
- Added CLI (`riskmetrics.cli`) using `argparse` for reproducible runs.
- Initialized Git repo and pushed to GitHub.

### Pitfalls & Best Practices
- Use `.venv` per project to avoid dependency conflicts.
- Prefer editable install: `pip install -e .` + `pyproject.toml` (no `PYTHONPATH` needed).
- Ignore generated folders via `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.vscode/`.
- Risk convention baseline: `loss = -pnl` (compute VaR/ES on losses).
- “git은 파일 변경 추적”, “pytest는 코드 검증” 역할이 다름.

### How to run
```bash
source .venv/bin/activate
pip install -e .
pytest -q
python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99
```

### Next
- Implement rolling risk metrics (rolling VaR) and add tests.

---

## 2026-03-04 — Pandas time series / rolling / alignment

### Context
- Learn pandas time-series workflow and implement rolling Historical VaR end-to-end (function + CLI + tests).
- 리스크 지표를 “시계열로 매일 갱신(rolling)”하는 흐름을 이해.

### Implementation
- Time-series basics:
  - `pd.to_datetime()`, `sort_values("date")`, `set_index("date").sort_index()`
- Implemented rolling Historical VaR:
  - `rolling_historical_var(pnl: pd.Series, window: int, alpha: float) -> pd.Series`
  - Uses `loss = -pnl` and `rolling(...).quantile(alpha)` with `min_periods=window`
- Exposed rolling VaR via CLI:
  - Added `--rolling-window` and printed the last 5 rolling VaR values.
- Added pytest coverage:
  - index/length preservation + NaN prefix
  - numeric check vs manual quantile reference (`np.quantile`)

### Pitfalls & Best Practices
- Rolling is order-sensitive: always sort before rolling (`set_index(...).sort_index()`).
- Keep pandas index for safe alignment; avoid early `.to_numpy()` in comparison logic.
- Use `pd.concat([...], axis=1).dropna()` to align series before comparisons.
- Small windows are noisy for VaR; realistic windows are typically larger (e.g., ~250 trading days).
- dropna를 너무 빨리 하면 길이/인덱스가 달라져 비교가 꼬일 수 있음.

### How to run
```bash
python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99 --rolling-window 3
pytest -q
```

### Next
- Add rolling violations + backtest summary (violation rate, Kupiec POF + p-value).
- Improve CLI to support a date column (print dates instead of row indices).

---

## 2026-03-05 — Backtesting pipeline: violations + Kupiec POF (LR + p-value)

### Context
- Implemented and validated the first backtesting component for rolling VaR.
- Goal was a reproducible pipeline: rolling VaR → violations → Kupiec POF test.

### Implementation
- Added `var_violations(loss: pd.Series, var: pd.Series) -> pd.Series`
  - Uses `pd.concat(...).dropna()` to align indices safely.
  - Defines violations as `loss > var`.
- Added `kupiec_pof_test(violations, alpha) -> dict`
  - Computes LR statistic under Binomial model and returns p-value via `chi2(df=1)`.
  - Added numerical stability guard (`eps`) to avoid `log(0)` when `x=0` or `x=n`.
- Verified end-to-end on `data/sample_pnl.csv` with `alpha=0.99`, `window=3`.

### Pitfalls & Best Practices
- Rolling/backtesting is order-sensitive → ensure time index sorted before rolling.
- Avoid early `.to_numpy()` in comparisons → keep pandas index alignment. (What we learned yesterday!)
- `x=0` 또는 `x=n`이면 `log(0)` 문제가 생겨서 작은 값 eps clipping이 필요함. (e.g., `eps=1e-12`)

### How to run
```bash
python - << 'PY'
import pandas as pd
from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import var_violations, kupiec_pof_test

df = pd.read_csv("data/sample_pnl.csv")
df["date"] = pd.date_range("2026-01-01", periods=len(df), freq="D")
df = df.set_index("date").sort_index()

pnl = df["pnl"]
loss = -pnl
alpha = 0.99
window = 3

rvar = rolling_historical_var(pnl, window=window, alpha=alpha)
viol = var_violations(loss, rvar)
print(kupiec_pof_test(viol, alpha=alpha))
PY
```

### Next
- Run the backtesting pipeline on a longer series (>= 250 observations) with a realistic window (e.g., 250 trading days).
- Produce a compact backtest report: (n, x, observed vs expected violation rate, LR_POF, p-value).
- Add a diagnostic plot: loss vs rolling VaR with violation markers to visually validate alignment and threshold breaches.

---

## 2026-03-06 — Long-series backtesting: rolling VaR + Kupiec POF + diagnostics

### Context
- Extend the rolling VaR backtesting pipeline from a toy example to a long series (>= 250 obs).
- Goal: validate coverage using Kupiec POF and add diagnostics (compact report + plot).

### Implementation
- Generated a long synthetic daily PnL series with 1000 observations:
  - PnL ~ N(0, 0.01^2)
  - Saved to `data/long_pnl.csv`
- Ran the full pipeline on a realistic rolling window:
  - `loss = -pnl`
  - `rolling_historical_var(pnl, window=250, alpha=0.99)`
  - `violations = var_violations(loss, rvar)` (index-aligned via `concat(...).dropna()`)
  - `kupiec_pof_test(violations, alpha=0.99)` → LR + p-value
- Added a compact helper: `backtest_report(loss, var, alpha)` returning:
  - n, x, expected_rate, observed_rate, lr_pof, p_value
- Created a diagnostic plot script:
  - `scripts/plot_backtest.py` plots loss vs rolling VaR with violation markers.

### Pitfalls & Best Practices
- Rolling reduces usable sample size: effective n = N - (window - 1).
  - With `alpha=0.99`, `window=250`, synthetic long series produced: n = 751, x = 9
- Keep pandas indices for correct alignment; use `concat(...).dropna()` before comparisons.
- Notebook import errors can occur if the kernel holds an old module version:
  - fix by restarting kernel / re-running imports.
- window가 250이니까 `n =1000 - 250 + 1 = 751` 확인함.

### How to run
```bash
# generate long series
python - << 'PY'
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 1000
pnl = rng.normal(0.0, 0.01, n)
pd.DataFrame({"pnl": pnl}).to_csv("data/long_pnl.csv", index=False)
print("Wrote data/long_pnl.csv with n =", n)
PY

# run backtest quickly
python - << 'PY'
import pandas as pd
from riskmetrics.var import rolling_historical_var
from riskmetrics.backtest import var_violations, kupiec_pof_test

alpha, window = 0.99, 250
df = pd.read_csv("data/long_pnl.csv")
df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
df = df.set_index("date").sort_index()

pnl = df["pnl"]
loss = -pnl
rvar = rolling_historical_var(pnl, window=window, alpha=alpha)
viol = var_violations(loss, rvar)
print(kupiec_pof_test(viol, alpha=alpha))
PY

# plot diagnostics
python scripts/plot_backtest.py
```

### Next
- Replace synthetic PnL with ar real price/return series (>=250 obs), then rerun report + diagnostics.
- Improve plotting for reporting: zoom recent window, save figure, and embed in notebook/README.
- Prepare the groundwork for returns/volatility featrues and historical VaR/ES functions.

---



