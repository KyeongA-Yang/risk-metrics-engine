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

---

## 2026-03-05 — Rolling VaR violations + Kupiec POF on sample PnL

### Question
Does the rolling VaR threshold achieve the expected coverage rate (violation probability = $1 - \alpha$)?

### Data
- `data/sample_pnl.csv` (10 observations)
- Added a synthetic daily date index for readability

### Setup
- `alpha = 0.99`
- `window = 3`
- `loss = -pnl`
- `violations = {loss > rolling_VaR}`
- Kupiec POF LR test with `chi-square(1)` p-value

### Results
- aligned `n = 8` (after rolling NaNs removed)
- violations `x = 2`
- `observed rate = 0.25`
- `expected rate = 0.01`
- `p-value ≈ 0.002`
- `Reject H0`

### Interpretation
- Reject H0 : The observed violation rate is far above the expected 1% for VaR(0.99).
- This strongly rejects correct coverage in this toy example.
- Due to small n and tiny window, results are for pipeline sanity check, not model assessment.

### Next
- Run on a longer series (>= 250) with realistic window sizes.
- Add summary reporting and plotting (loss vs rolling VaR, violation markers).

### Summary
- 파이프라인이 정상 작동됨을 확인하였고, 샘플 데이터(window=3)에서 위반률이 기대치(1%)보다 훨씬 커서 Kupiec POF에서 기각되는 결과도 함께 확인하였다.

---

## 2026-03-06 — Long-series rolling VaR backtest (Kupiec POF + diagnostics)

### Question
Does rolling historical VaR achieve correct coverage on a long series (violation probability ≈ 1 - alpha)?

### Data
- Synthetic daily PnL series of length N=1000
- Generated from Normal(0, 0.01^2) and saved as `data/long_pnl.csv`
- Added a synthetic daily date index for alignment and plotting

### Setup
- `alpha = 0.99` (expected violation rate = 0.01)
- `rolling window = 250`
- `loss = -pnl`
- rolling VaR computed via empirical quantile on loss
- violations defined as `loss > rolling_VaR`
- Kupiec POF LR test with `chi-square(1)` p-value

### Results
- Effective `n = 751` (after removing rolling NaNs)
- Violations `x = 9`
- `Observed rate = 0.011984`
- `Expected rate = 0.01`
- `LR_POF = 0.2808`
- `p_value = 0.5962`

### Interpretation
- We fail to reject correct coverage at common significance levels (e.g., 5%).
- Observed violation rate is close to the expected 1% for VaR(0.99).
- This run is primarily a sanity check of the end-to-end pipeline on a long series.
- Diagnostics
  - Visual check: plotted loss vs rolling VaR with violation markers.
  - Confirmed violations occur only when loss exceeds the VaR threshold.

### Next
- Run the same pipeline on a real return series and compare results across windows/alpha values.
- Save and embed plots for reporting.
- Add volatility features (rolling vol) to analyze how risk changes over regimes.

### Summary
- 긴 시계열(N=1000)에서 rolling VaR 백테스트를 수행했고, 위반률이 기대치(1%)와 크게 다르지 않아 Kupiec POF에서 기각되지 않으며 파이프라인이 정상 동작함을 확인했다.

---

## 2026-03-09 — SPY long-series backtest (rolling historical VaR, Kupiec POF, diagnostics)

### Question
Does rolling historical VaR achieve correct coverage (violation probability ≈ $1 - \alpha$) on real market data (SPY)?

### Data
- SPY daily adjusted price series (5y) downloaded via `yfinance`
- Converted to daily log returns:
  - `pnl_t = log(price_t) - log(price_{t-1})`
  - `loss_t = -pnl_t`
- Length: 1256 prices → 1255 returns

### Setup
- `alpha = 0.99` (expected violation rate = 0.01)
- `rolling window = 250`
- Rolling historical VaR via empirical quantile within window
- Violations: `I_t = 1{ loss_t > VaR_{alpha,t} }`
- Kupiec POF LR test with `chi-square(1)` p-value
- Diagnostic plot: loss vs rolling VaR with violation markers (zoom last 300), plus a “loss-only” version for readability

### Results
- Effective `n = 1006`
- Violations `x = 19`
- `Observed rate = 0.0188867`
- `Expected rate = 0.01`
- `LR_POF = 6.3636`
- `p_value = 0.01165`

### Interpretation
- Reject correct coverage at 5% significance: the observed violation rate is higher than expected.
- Suggests under-coverage for 99% historical VaR on SPY in this period.
- This is consistent with real-market behavior (heavy tails / volatility clustering) where simple historical VaR can be challenged.
- Diagnostics
  - Report-ready plot (loss-only) improves readability and confirms that violations occur exactly when loss exceeds the rolling VaR threshold.
  - Saving plots as PNG supports reproducible reporting on GitHub.

### Next
- Compare coverage across (alpha, window) settings and summarize how results change.
- Evaluate the impact of out-of-sample shift (t-1 estimation) on violation rate and Kupiec POF.
- Extend beyond coverage:
  - rolling ES
  - independence / clustering diagnostics (e.g., Christoffersen-style ideas)

### Summary
- SPY 실데이터(5년)에서 rolling historical VaR(99%, window=250)의 위반률이 기대(1%)보다 높아 Kupiec POF에서 기각(p≈0.0116)되었고, plot을 통해 위반이 임계값 초과 구간에서 발생함을 시각적으로 확인했다.

---

## 2026-03-10 — Extreme-loss classifier baseline (SPY): threshold sweep + probability diagnostics

### Question
Can a simple baseline (logistic regression) predict next-day extreme loss events from basic return/volatility/VaR features?

### Data
- SPY daily prices (5y) already saved as `data/price_SPY.csv`
- Converted to log returns and loss:
  - $r_t = \log(P_t) - \log(P_{t-1})$
  - $L_t = -r_t$

### Setup
- Features: `ret`, `vol20`, `var250`
- Label:
  - `loss_t1 = L_{t+1}`
  - Train-only threshold: $\mathrm{thr} = q_{\mathrm{label\_q}}(L_{t+1})$
  - $y_t = 1\{ L_{t+1} > \text{thr} \}$
- Split: time-based 80/20 (no shuffle)
- Model: `StandardScaler + LogisticRegression`
- Evaluation:
  - proba summary + quantiles
  - mean proba by class
  - ROC-AUC
  - threshold sweep for precision/recall trade-off

### Results
- With `label_q=0.95`, test positives were extremely small (often 0–2), making metrics unstable.
- With `label_q=0.90`, test positives increased (e.g., 8/201), but:
  - predicted probabilities had a narrow range
  - `mean proba(y=1)` was not higher than `mean proba(y=0)` in this run
  - threshold sweep showed extreme behavior: either almost no positives predicted or almost all predicted as positives depending on threshold

### Interpretation
- The baseline does not separate rare events well with the current feature set.
- Probability diagnostics explain why thresholding can fail:
  - narrow proba distribution → small threshold changes cause large behavior changes
  - if `proba(y=1)` is not higher than `proba(y=0)`, ROC-AUC can fall below 0.5

### Next
- Add lagged returns (recent-day patterns) and compare:
  - proba separation, ROC-AUC, and threshold sweep behavior
- Consider alert-budget evaluation (e.g., top-K highest proba days) instead of fixed thresholds.

### Summary
- SPY 실데이터 기반으로 내일의 큰 손실 분류 baseline을 만들고, threshold sweep과 확률 진단을 통해 Feature 신호가 약하면 모델이 분리되지 않는 현상을 확인했다.
- 현재 baseline 모델에서는 test 구간에서 위험일을 정상일보다 높은 점수로 랭킹하지 못했고, 모델의 분리력이 부족했음.

---

## 2026-03-11 — Coverage grid on SPY: alpha × window (rolling historical VaR)

### Question
How does coverage change across $(\alpha, \text{window})$ settings for rolling historical VaR on SPY?

### Data
- SPY daily prices (5y) from `data/price_SPY.csv`
- Returns and losses:
  - $r_t = \log(P_t) - \log(P_{t-1})$
  - $L_t = -r_t$

### Setup
- Grid:
  - $\alpha \in \{0.975, 0.99\}$
  - window $\in \{125, 250, 500\}$
- Rolling historical VaR computed from pnl/returns within each window.
- Violations:

$$ I_t = \mathbf{1}_{\{L_t > \mathrm{VaR}_{\alpha,t}\}} $$
  
- Summary per setting via `backtest_report`:
  - $n$ (valid days), $x$ (violations), expected rate $(1-\alpha)$, observed rate $(x/n)$
  - Kupiec POF LR statistic and p-value

### Results (from coverage grid)
- $\alpha=0.975$:
  - window 125: observed rate ≈ 0.0407 > 0.025, p ≈ 0.0019 (reject)
  - window 250: observed rate ≈ 0.0328 > 0.025, p ≈ 0.13 (no reject)
  - window 500: observed rate ≈ 0.0251 ≈ 0.025, p ≈ 0.98 (matches well, no reject)
- $\alpha=0.99$:
  - window 125: observed rate ≈ 0.0212 > 0.01, p ≈ 0.0010 (reject)
  - window 250: observed rate ≈ 0.0189 > 0.01, p ≈ 0.0116 (reject)
  - window 500: observed rate ≈ 0.00794 < 0.01, p ≈ 0.55 (no reject)

### Interpretation
- Short windows (125, 250) tend to under-cover, especially at $\alpha=0.99$ (too many violations).
- A longer window (500) stabilizes quantile estimation and yields coverage closer to the expected rate.
- This illustrates the bias–variance trade-off in rolling historical quantile estimation.

### Next
- Add a 1-step out-of-sample shift: compare $L_t$ to $\mathrm{VaR}_{\alpha,t-1}$ to avoid look-ahead.
- Extend beyond coverage: rolling ES and independence/clustering diagnostics.

### Summary
- SPY 실데이터에서 rolling historical VaR의 coverage를 $(\alpha, window)$ grid로 비교했고, 짧은 window에서 특히 99% VaR가 under-coverage(위반 과다)로 Kupiec POF에서 기각되는 패턴을 확인했다.

---

## 2026-03-16 — ML: extreme-loss prediction with lagged returns (Logit vs RF + alert-budget)

### Question
Do lagged return features improve next-day extreme-loss prediction, and does a non-linear baseline (RandomForest) outperform logistic regression under rare-event evaluation?

### Data
- SPY daily prices from `data/price_SPY.csv`
- Returns/loss:
  - $r_t = \log(P_t) - \log(P_{t-1})$
  - $L_t = -r_t$

### Setup
- Feature set at time $t$:
  - `ret`, `vol20` (20-day rolling std), `var250` (250-day rolling 1% quantile)
  - lag features: `ret_lag1`…`ret_lag5`
- Label:
  - `loss_t1(t) = loss(t+1)`
  - Train-only threshold: `thr = q_{label_q}(train.loss_t1)`
  - $y_t = 1\{L_{t+1} > \mathrm{thr}\}$
- Split:
  - chronological 80/20 train/test (no shuffle)
- label_q:
  - `label_q = 0.90` (top 10% tomorrow-loss days as positives)
- Models:
  - Logistic regression (scaled, class_weight balanced)
  - RandomForest (balanced_subsample)
- Metrics:
  - ROC-AUC (ranking)
  - Threshold sweep (precision/recall)
  - Alert-budget: Recall@K for `K ∈ {5,10,20,50}`

### Results (label_q=0.90)
- Class balance:
  - train positives: 81 / 804 (0.1007)
  - test positives: 8 / 201 (0.0398)
- Logistic regression:
  - ROC-AUC ≈ 0.453
  - Alert-budget:
    - Recall@5 = 0.00
    - Recall@10 = 0.00
    - Recall@20 = 0.00
    - Recall@50 = 0.25
- RandomForest:
  - ROC-AUC ≈ 0.694
  - Alert-budget:
    - Recall@5 = 0.00
    - Recall@10 = 0.125
    - Recall@20 = 0.125
    - Recall@50 = 0.50

### Interpretation
- Logistic regression shows weak ranking ability (AUC < 0.5) and poor top-K capture at small K.
- RandomForest improves ranking (AUC ~ 0.69) and captures more extreme days under an alert budget (Recall@50 ~ 0.50).
- Suggests non-linear interactions among lagged returns and volatility/state features may matter.
- Test positives are few (n=8), so top-K metrics can be noisy; results should be confirmed with additional periods or walk-forward validation.

### Next
- Add Precision@K and plot precision–recall trade-offs for alerting.
- Inspect feature importance (RF) to understand which lags/features drive performance.
- Try simpler/non-leaky state features (e.g., rolling vol only) and compare robustness.
- Consider walk-forward / rolling window evaluation to reduce sensitivity to a single split.

### Summary
- lag features를 추가했을 때 Logit보다 RF가 AUC와 Recall@K에서 더 좋은 성능을 보여서, 비선형 패턴(상호작용)이 존재할 가능성을 확인했다. (그래도 정확도는 그렇게 높진 않음.)

---

## 2026-03-16 — Same-day vs OOS (1-step shift) VaR backtest on SPY + coverage grid

### Question
How different are same-day and out-of-sample (1-step shift) VaR coverage results?
Does window length improve coverage on real market data (SPY)?

### Data
- SPY daily prices (5y) → log returns $r_t$ → loss $L_t = -r_t$
- Also tested a synthetic long PnL series for sanity check.

### Setup
- Grid:
  - $\alpha \in \{0.975, 0.99\}$
  - window $\in \{125, 250, 500\}$
- Same-day violations:


  $$I_t^{\mathrm{same}} = \mathbf{1}_{\{L_t > \mathrm{VaR}_{\alpha,t}\}}$$


- OOS violations:


  $$I_t^{\mathrm{oos}} = \mathbf{1}_{\{L_t > \mathrm{VaR}_{\alpha,t-1}\}}$$


### Results
- Short windows (125) show clear under-coverage (observed violation rate > expected), often rejected by Kupiec POF.
- Longer windows (500) move observed rates closer to expected and are less likely to be rejected.
- Same-day vs OOS differences were small in most settings (OOS uses 1 fewer observation), but OOS is preferred for leakage-safe evaluation.

### Interpretation
- Window length strongly affects tail quantile stability.
- Real returns exhibit heavy tails / volatility clustering, challenging high-confidence (99%) historical VaR.

### Next
- Plot the grid summary (observed vs expected) for report-ready comparison.
- Extend beyond coverage: rolling ES and independence/clustering diagnostics.

### Summary
- 같은 날의 손실이 VaR 계산에 들어가면 그 VaR로 다시 손실을 평가하는 구조라서, out-of-sample이 더 정석적이다. (결과 자체는 살짝만 달라짐.)
- SPY에서 작은 window에서는 under-coverage가 나타나서 POF가 기각되기 쉽고, window를 키우면 안정화되면서 크게 개선된다.

---

## 2026-03-17 — ML diagnostics: alert-budget metrics + expanding walk-forward evaluation

### Question
1) With a fixed daily alert budget (Top-K alerts), how well do we capture next-day extreme-loss events?  
2) Does performance look stable across time under an expanding walk-forward evaluation (vs a single 80/20 split)?

### Setup (only what’s new/important)
- Added **alert-budget metrics** to evaluation:
  - Precision@K and Recall@K for $K \in \{5,10,20,50\}$
- Added a more realistic **time-series evaluation design**:
  - **Expanding walk-forward** splits (train grows over folds; test is the next block)
  - **Fold-specific, train-only thresholding** to avoid leakage:
    - $\mathrm{thr}^{(\mathrm{fold})} = q_{\mathrm{label\_q}}\left(\{L_{t+1}: t \in \mathrm{train\ fold}\}\right)$
    - $y_t^{(\mathrm{fold})} = \mathbf{1}_{\{L_{t+1} > \mathrm{thr}^{(\mathrm{fold})}\}}$

### Results (high-level)
- Single-split results can be misleading for rare events (positives are few / thresholds are sensitive).
- Walk-forward diagnostics showed:
  - thresholds vary by fold (regime-dependent tails),
  - metrics vary across folds (regime dependence / instability).

### Interpretation
- **Precision@K / Recall@K** are more deployment-relevant than a fixed probability cutoff:
  - Precision@K reflects false-alarm burden under a daily alert budget.
  - Recall@K reflects how many extreme days we catch with limited alerts.
- **Expanding walk-forward** is preferred for leakage-safe time-series ML reporting:
  - repeated “train on past → test on future” evaluation,
  - exposes regime shifts that a single split can hide.

### Next
- Summarize walk-forward results in a compact table (mean/std across folds):
  - ROC-AUC, Precision@K, Recall@K
- Add probability calibration (Platt / isotonic) and evaluate calibration quality.
- Compare expanding vs rolling walk-forward (fixed-length training) for robustness.

### Summary
- alert-budget 지표(Precision@K/Recall@K)와 expanding walk-forward 평가를 추가해, 단일 split보다 현실적인 방식으로 알림 예산 관점 성능 + 시간에 따른 성능 변동을 점검할 기반을 만들었다.

---