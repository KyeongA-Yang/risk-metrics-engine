# Engineering Log
> Template: Context → Implementation → Pitfalls & Best Practices → How to run → Next

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

