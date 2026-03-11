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
cd ~/projects/risk-metrics-engine
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

## 2026-03-09 — Real-data backtesting: SPY (price → returns) + Kupiec POF + report-ready plot

### Context
- Replace synthetic PnL with a real market series (>=250 obs) and rerun the backtesting pipeline.
- Goal: produce a compact backtest report and a report-ready diagnostic plot.
- Also reduce code duplication by generalizing the plotting script.

### Implementation
- Downloaded SPY daily prices (5y, auto-adjusted) using `yfinance` and saved a clean CSV:
  - `data/price_SPY.csv` with columns: `date`, `price`
- Converted price to log returns and loss:
  - `pnl_t = log(price_t) - log(price_{t-1})`
  - `loss_t = -pnl_t`
- Reran rolling historical VaR backtest:
  - `alpha = 0.99`, `window = 250`
  - `rolling_historical_var(pnl, window, alpha)`
  - `backtest_report(loss, VaR, alpha)` → compact dict: (n, x, expected/observed rate, LR_POF, p-value)
- Improved diagnostics for reporting:
  - Plot only positive losses: `loss_pos = clip(loss, lower=0)`
  - Zoom recent window (e.g., last 300 points)
  - Save figure to PNG: `data/spy_backtest_plot_loss_only.png`
- Refactored plotting into a single generalized script (`scripts/plot_backtest.py`) with CLI args:
  - `--mode pnl` (synthetic pnl CSV) vs `--mode price` (date+price CSV)
  - `--zoom-last`, `--loss-only`, `--out` for report-ready output
- 실제 S&P500을 이용하여 backtest를 실행해보았고, loss(0 이상인 값만 나오도록)의 꼬리 부분만 확대하여 좀 더 보기 편한 plot으로 그려보았다.

### Pitfalls & Best Practices
- Results: 
  - Observed rate ≈ 0.01889 vs expected rate = 0.01
  - Kupiec POF p-value ≈ 0.01165 → reject correct coverage at 5% (more violations than expected)
- Real market returns exhibit heavy tails and volatility clustering; simple historical VaR can under-cover at high confidence levels (e.g., 99%).
- Keep index alignment explicit using `pd.concat([...], axis=1).dropna()` before comparisons.
- Prefer report-ready plots (zoom + save) for GitHub/portfolio reproducibility.
- 금융 실데이터는 꼬리가 두껍고, 변동성이 존재하여서 단순 historical window가 99% tail을 잘 못 맞추는 경우가 흔함.

### How to run
```bash
# (1) Download SPY prices (once)
python - << 'PY'
import yfinance as yf, pandas as pd
ticker = "SPY"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
out = df[["Date", "Close"]].rename(columns={"Date":"date","Close":"price"})
out.to_csv("data/price_SPY.csv", index=False)
print("Saved data/price_SPY.csv rows:", len(out))
PY

# (2) Run generalized diagnostic plot (SPY)
python scripts/plot_backtest.py --csv data/price_SPY.csv --mode price --alpha 0.99 --window 250 --zoom-last 300 --loss-only --out data/plot_spy_loss_only.png
```

### Next
- Add a small coverage grid runner (alpha × window) and print a compact table of results.
  - Example: alpha = 0.975, 0.99; window = 125, 250, 500
- Upgrade to an out-of-sample backtest:
  - compute VaR from data up to t-1 (1-step shift) before comparing to L_t
- Prepare utilities for price → returns → volatility features,
  then add historical VaR/ES functions based on returns Series.

---

## 2026-03-10 — ML baseline (time-series, leakage-safe): extreme-loss classifier + diagnostics

### Context
- Start ML track (Tue/Thu) while keeping the Risk Metrics Engine repo as the main portfolio project.
- Build a minimal, leakage-safe baseline to predict next-day “extreme loss” events from SPY returns.

### Implementation
- Data pipeline (real market series):
  - Loaded `data/price_SPY.csv` (SPY daily prices) and computed log returns and losses:
    - $r_t = \log(P_t) - \log(P_{t-1})$
    - $L_t = -r_t$
  - Built features:
    - `ret_t`
    - `vol20_t = rolling std(ret, 20)`
    - `var250_t = rolling quantile(ret, 0.01, 250)`
  - Built label using next-day loss:
    - `loss_t1 = shift(loss, -1)` so `loss_t1(t) = loss(t+1)`

- Train-only threshold to avoid leakage:
  - Threshold: $\mathrm{thr} = q_{\mathrm{label\_q}}\left(\{L_{t+1}: t \in \mathrm{train}\}\right)$
  - Label:$y_{t} = \mathbf{1}_{L_{t+1} > \mathrm{thr}}$


- Modeling:
  - Time-based split 80/20 (no shuffle)
  - Baseline classifier: `StandardScaler + LogisticRegression` in a pipeline
  - Added diagnostics and threshold sweep:
    - proba distribution summary / quantiles
    - mean proba by class
    - ROC-AUC (only if both classes exist in test)
    - threshold sweep table: TP/FP/FN/TN, precision/recall

### Pitfalls & Best Practices
- Use train-only preprocessing parameters:
  - `StandardScaler` learns $$\mu,\sigma$$ on train; applies same $$\mu,\sigma$$ to test (prevents leakage).
- Rare-event labels can lead to unstable evaluation when test positives are very small.
- Avoid relying on accuracy; inspect precision/recall and probability diagnostics.
- If proba distribution is narrow and `mean proba(y=1) ≤ mean proba(y=0)`, the baseline is not separating the event.

### How to run
```bash
python scripts/ml_extreme_loss_baseline.py
```

### Next
- Add lagged return features (e.g., ret_lag1…ret_lag5) and re-run diagnostics.
- Consider alert-budget evaluation for practical risk alerting.
- Compare logistic baseline vs a simple non-linear model (e.g., RandomForest) once features are ready.

  ---

## 2026-03-11 — Coverage grid runner (SPY): rolling VaR backtest table

### Context
- Extend the risk backtesting workflow beyond a single setting by running a small grid over $(\alpha, \text{window})$.
- Goal: produce a compact, reproducible table that summarizes coverage and Kupiec POF results on real data (SPY).

### Implementation
- Prepared a real-data loss series from SPY prices:
  - $r_t = \log(P_t) - \log(P_{t-1})$
  - $L_t = -r_t$
- Implemented a coverage grid runner (script + notebook workflow):
  - Grid: `alpha ∈ {0.975, 0.99}`, `window ∈ {125, 250, 500}`
  - For each pair:
    - compute `rolling_historical_var(pnl, window, alpha)`
    - run `backtest_report(loss, VaR, alpha)` to get a compact dict:
      - `(n, x, expected_rate, observed_rate, LR_POF, p_value)`
- Saved results for reporting:
  - Printed a DataFrame table in notebook/terminal
  - Exported `data/coverage_grid_spy.csv`

### Pitfalls & Best Practices
- Use explicit loss convention: `loss = -pnl` and compare `loss > VaR`.
- Ensure time ordering: sort by date and keep index alignment (avoid early `.to_numpy()`).
- Reporting-friendly output: a single table makes comparisons across settings easy.
- 짧은 window에서는 tail(극단 구간) 추정이 불안정해서 위반률이 커질 수 있음.

### How to run
```bash
python scripts/run_coverage_grid.py
```

### Next
- Upgrade to an out-of-sample backtest using a 1-step shift:
  - compare $L_t$ to $\mathrm{VaR}_{\alpha,t-1}$ (estimated from past only)
- Add a compact plot/report template to summarize coverage across the grid.

---

