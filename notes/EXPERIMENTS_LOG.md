# Experiments Log
> Template: Question → Data → Setup → Results → Interpretation → Next → Summary

## 2026-03-03 — End-to-end sanity run (VaR/ES CLI)

### Question
- Can the package run end-to-end (install → CLI → VaR/ES output) on a simple sample PnL CSV?

### Data
- File: `data/sample_pnl.csv`
- Column: `pnl` (PnL values)

### Setup
- Environment: `.venv` (Python 3.12)
- CLI command:
  - `python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99`
- Metrics computed:
  - Historical VaR (empirical quantile on loss)
  - Parametric VaR (Normal assumption on loss)
  - Historical ES (tail mean on loss)

### Results
- CLI successfully printed VaR/ES values for `alpha=0.99`.
- Confirmed the sign convention: computed on `loss = -pnl` and reported VaR/ES as positive loss magnitudes.

### Interpretation
- The pipeline works: input CSV → parse args → compute metrics → print results.
- This provides a reproducible baseline for adding rolling metrics and backtesting.

### Next
- Implement rolling historical VaR and add numeric validation tests.

### Summary
- 샘플 PnL로 CLI가 VaR/ES를 정상 출력하는지 end-to-end로 확인했고, `loss = -pnl` 부호 규칙을 확정했다.

---

## 2026-03-04 — Rolling Historical VaR + alignment checks

### Question
- Does rolling historical VaR behave as expected, and do index-alignment choices affect correctness?

### Data
- Primary: `data/sample_pnl.csv` (n=10)
- Demo time index: created via `pd.date_range(...)` in notebooks (since sample CSV has no date column)

### Setup
- Rolling VaR definition (loss-based):
  - `loss = -pnl`
  - `VaR_{α,t} = q_α(loss_{t-w+1:t})`
- Parameters:
  - `alpha = 0.99`
  - `window = 3` (demo; realistic windows are typically much larger, e.g., ~250 trading days)
- CLI command:
  - `python -m riskmetrics.cli --csv data/sample_pnl.csv --alpha 0.99 --rolling-window 3`
- Alignment experiments:
  - Compared pandas Series alignment vs dropping index via `.to_numpy()`
  - Used `pd.concat([...], axis=1).dropna()` to align safely

### Results
- Rolling VaR printed last 5 values successfully for `window=3`.
- Observed high variability in rolling VaR due to a very small window.
- Demonstrated that `.to_numpy()` + `dropna()` can cause shape mismatch errors (or silent misalignment if lengths happen to match).

### Interpretation
- With small window sizes, high quantiles (0.99) are unstable and can approximate the window maximum.
- Correctness depends on maintaining time order and index alignment:
  - sort before rolling
  - keep Series indices during comparisons
  - align explicitly with `concat(...).dropna()` when necessary

### Next
- Add rolling violations and backtest summary:
  - violation rate and Kupiec POF (with p-value)
- Improve CLI to support a date column and print dates instead of row indices.

### Summary
- rolling VaR를 함수/CLI로 구현해 동작을 확인했고, `dropna`/`.to_numpy()`로 인덱스 정합이 깨질 수 있어 `concat(...).dropna()`로 맞추는 게 안전하다는 걸 확인했다.