# Risk Metrics Engine (Python)

A small Python package to compute basic risk metrics from PnL data:
- Historical VaR / Expected Shortfall (ES)
- Rolling Historical VaR
- Basic backtesting helpers (VaR violations, Kupiec POF statistic)

## Conventions
- Input data is **PnL** (Profit and Loss): `pnl > 0` profit, `pnl < 0` loss
- We convert to **Loss** via: `loss = -pnl`
- VaR/ES are reported as **positive loss magnitudes** (e.g., `0.03` = 3% loss threshold)

## Quickstart

### 1) Setup (venv)
Create and activate a virtual environment, then install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Run tests
Run the test suite:

```bash
pytest -q
```

### 3) Run CLI
Compute single-number VaR/ES:

```bash
python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99
```

Compute rolling historical VaR (demo with a small window):

```bash
python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99 --rolling-window 3
```

## Usage

### Input CSV format
CSV must contain a PnL column (default: `pnl`).

Example:
```csv
pnl
0.015
-0.020
...
```

### CLI options
- `--csv` (required): path to CSV
- `--col` (default: `pnl`): column name for PnL
- `--alpha` (default: `0.99`): confidence level for VaR/ES
- `--rolling-window` (optional): rolling window size for rolling historical VaR

### Library usage (Python)
```python
import pandas as pd
from riskmetrics.var import historical_var, rolling_historical_var
from riskmetrics.es import historical_es

df = pd.read_csv("data/sample_pnl.csv")

pnl_np = df["pnl"].to_numpy()
pnl_series = df["pnl"]

v = historical_var(pnl_np, alpha=0.99)
es = historical_es(pnl_np, alpha=0.99)
rv = rolling_historical_var(pnl_series, window=250, alpha=0.99)
```

## Project structure
- `src/riskmetrics/` : core package code
- `tests/` : pytest suite (includes numeric checks)
- `scripts/` : runnable scripts
- `notebooks/` : exploratory notebooks
- `data/` : sample input data

## Roadmap (ideas)
- Rolling ES
- VaR backtesting report (violation rate, Kupiec test p-value)
- Plotting: loss series + rolling VaR + violation markers
- Support date column in CLI output (print dates instead of row indices)


## Learning Notes

### 2026-03-04 — Rolling / time series basics
- `pd.to_datetime()`은 날짜/시간을 파이썬이 **datetime 타입**으로 인식하게 만든다.
- `sort_values("date")`는 시간 순서가 꼬인 데이터를 **날짜 기준으로 정렬**한다.
- `set_index("date")`를 하면 날짜가 **인덱스**가 되어 시계열 연산(rolling 등)이 자연스러워진다.
- `rolling(window=3)`은 “최근 3개 관측치” 기준으로 계산하며, 초반 `window-1` 구간은 값이 없어 `NaN`이 생긴다.
- rolling 결과는 원래 데이터와 **길이/인덱스가 같게 유지**되는 것이 기본이라 “정렬/정합성(alignment)”이 중요하다.

### 2026-03-04 — dropna / alignment pitfalls
- `dropna()`는 `NaN`이 있는 행을 제거해서 “계산 가능한 구간만” 남긴다.
- rolling 결과는 초반에 `NaN`이 생기므로, 필요하면 `dropna()`로 제거할 수 있다.
- pandas는 Series 연산 시 “행 순서”가 아니라 “인덱스(날짜)”로 **자동 정렬(alignment)** 해서 계산한다.
- 그래서 `dropna()`로 한쪽 인덱스가 줄면 비교/연산 결과가 예상과 달라질 수 있다.
- 안전한 방식: `set_index(...).sort_index()` 후 `pd.concat([...], axis=1).dropna()`로 **공통 인덱스만 맞춘 뒤** 비교한다.
