# Risk Concepts
> Definitions, formulas, intuition, and pitfalls (VaR/ES/Backtesting).

This note defines core risk metrics and backtesting concepts used in this project.
**Convention used throughout:**  
- PnL: profit and loss (`pnl > 0` profit, `pnl < 0` loss)  
- Loss: $L = -\mathrm{PnL}$  
- VaR/ES are reported as **positive loss magnitudes**.

---

## 0) Notation
- $\mathrm{PnL}_t$: profit and loss at time $t$
- $L_t = -\mathrm{PnL}_t$: loss at time $t$
- $\alpha \in (0,1)$: confidence level (e.g., $0.99$)
- $q_\alpha(X)$: $\alpha$-quantile of random variable $X$
- Window size $w$: number of observations in rolling estimation (e.g., 250 trading days)

---

## 1) Quantile (VaR building block)
### Definition
For a random variable $X$, the $\alpha$-quantile is:
$$
q_\alpha(X) = \inf \{x \in \mathbb{R} : \mathbb{P}(X \le x) \ge \alpha\}.
$$

### Practical note (empirical quantile)
For finite samples, quantiles depend on the chosen method (interpolation rule).  
With very small sample sizes, high quantiles (e.g., 0.99) can behave like “almost the maximum”.

---

## 2) Value at Risk (VaR)
### Definition (loss-based)
$$
\mathrm{VaR}_\alpha = q_\alpha(L).
$$

### Interpretation
$$
\mathbb{P}(L > \mathrm{VaR}_\alpha) \approx 1-\alpha.
$$
Example: If $\alpha=0.99$ and $\mathrm{VaR}_{0.99}=0.03$, then “loss exceeds 3% with probability about 1%”.

### Historical (nonparametric) VaR estimate
Given observed losses $\{L_1,\dots,L_n\}$:
$$
\widehat{\mathrm{VaR}}_\alpha = q_\alpha(\{L_i\}_{i=1}^n).
$$

### Pitfalls / notes
- VaR is a *threshold*, not a tail average; it does not describe severity beyond the cutoff.
- Quantile method matters for small samples.
- VaR is not subadditive in general (risk aggregation caveat).

---

## 3) Expected Shortfall (ES) / Conditional VaR (CVaR)
### Definition
$$
\mathrm{ES}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha].
$$

### Interpretation
ES answers: “If we are already in the worst $1-\alpha$ tail, what is the average loss?”

### Historical ES estimate (simple)
Let $\widehat{\mathrm{VaR}}_\alpha$ be the empirical VaR and define the tail set:
$$
\mathcal{T} = \{ i : L_i \ge \widehat{\mathrm{VaR}}_\alpha \}.
$$
Then
$$
\widehat{\mathrm{ES}}_\alpha = \frac{1}{|\mathcal{T}|} \sum_{i\in \mathcal{T}} L_i.
$$

### Pitfalls / notes
- Tail sample size can be tiny (especially at $\alpha=0.99$), leading to high variance.
- ES is generally considered a “coherent” risk measure under standard assumptions.

---

## 4) Rolling (time-varying) risk metrics
Rolling estimation recomputes the metric at each time $t$ using only the most recent window.

### Rolling VaR (loss-based)
$$
\mathrm{VaR}_{\alpha,t} = q_\alpha\left(L_{t-w+1}, \dots, L_t\right).
$$

### Rolling ES
$$
\mathrm{ES}_{\alpha,t} = \mathbb{E}\left[L \mid L \ge \mathrm{VaR}_{\alpha,t}\right] \quad \text{estimated within the same window.}
$$

### Practical notes
- **Order matters:** rolling assumes data are sorted by time.
- **Window choice:** small $w$ = noisy estimate; large $w$ = smoother, more stable estimate.
- For daily trading PnL, $w \approx 250$ is common as a “1-year” window.

---

## 5) Backtesting: VaR violations
Define the violation indicator:
$$
I_t = \mathbf{1}\{L_t > \mathrm{VaR}_{\alpha,t}\}.
$$

### Interpretation
If the VaR model is well-calibrated, then:
$$
\mathbb{E}[I_t] \approx 1-\alpha.
$$
So for $\alpha=0.99$, the expected violation rate is about 1%.

---

## 6) Kupiec POF (Proportion of Failures) test (basic)
Let $n$ be the number of backtest observations and $x=\sum_{t=1}^n I_t$ be the number of violations.  
Under the null hypothesis $H_0$: violation probability $p = 1-\alpha$.

Kupiec’s likelihood ratio statistic compares:
- $H_0$: $p$ fixed at $1-\alpha$
- $H_1$: $p$ estimated by $\hat p = x/n$

$$
\mathrm{LR}_{\text{POF}} = -2\left[\log L(p) - \log L(\hat p)\right],
$$
where $L(\cdot)$ is the binomial likelihood.

### Interpretation
- Large $\mathrm{LR}_{\text{POF}}$ suggests the observed violation rate is inconsistent with the expected $1-\alpha$.

### Practical note
To report significance, compute a p-value using $\chi^2$ with 1 degree of freedom:
$$
p\text{-value} = 1 - F_{\chi^2_1}(\mathrm{LR}_{\text{POF}}).
$$
(We can add this to the code later.)

---

## 7) Common implementation pitfalls (pandas)
- Rolling computations are order-sensitive → sort your time index first.
- Pandas aligns by index during Series operations (alignment). Avoid dropping index information too early (e.g., `.to_numpy()`).
- When comparing two Series (e.g., loss vs rolling VaR), prefer:
  - `pd.concat([loss, var], axis=1).dropna()` then compare columns.

---
