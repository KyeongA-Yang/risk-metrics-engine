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

## Documentation
- Engineering log: `notes/ENGINEERING_LOG.md`
- Experiments log: `notes/EXPERIMENTS_LOG.md`
- Risk concepts: `notes/RISK_CONCEPTS.md`
