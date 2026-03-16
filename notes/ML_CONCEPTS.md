# ML Concepts
> Definitions, formulas, intuition, and pitfalls for ML tasks.

This note defines core ML concepts used for time-series classification in this repo.

**Convention used throughout**
- We build features using information available at time $t$
- We predict an outcome at time $t+1$ (next-day event)
- All preprocessing parameters (e.g., scaling, label thresholds) are learned on train only

---

## 0) Notation
- Price: $P_t$
- Log return: $r_t = \log(P_t) - \log(P_{t-1})$
- Loss: $L_t = -r_t$
- Feature vector at time $t$: $X_t$
- Binary label at time $t$: $y_t \in \{0,1\}$
- Train/test split index: $T_{\text{split}}$
- Predicted probability (score): $\hat p_t = \mathbb{P}(y_t=1 \mid X_t)$
- Quantile operator: $q_\alpha(\cdot)$
- Label quantile (train-only): $\mathrm{label\_q} \in (0,1)$
- Extreme-loss threshold (train-only): $\mathrm{thr}$

---

## 1) Problem setup (next-day extreme-loss classification)

### Definition
We predict whether tomorrow is an extreme-loss day (a rare event):

$$ y_t = \mathbf{1}_{\{L_{t+1} > \mathrm{thr}\}} $$

### Interpretation
- $y_t=1$ means: “tomorrow ($t+1$) loss is unusually large”
- Features $X_t$ must use only information up to time $t$ (no future leakage)

### Practical note (implementation)
A common implementation uses a shift:
- `loss_t1 = loss.shift(-1)` so `loss_t1(t) = loss(t+1)`
- then define `y_t = 1{ loss_t1(t) > thr }`

### Pitfall
If you accidentally use future information when building $X_t$ (e.g., using `shift(-1)` inside features), evaluation becomes invalid.

---

## 2) Leakage and leakage-safe labeling

### Leakage-free thresholding (train-only)
The extreme-loss cutoff must be computed using the training period only:

$$ \mathrm{thr} = q_{\mathrm{label\_q}}\left(\{L_{t+1}: t \in \mathrm{train}\}\right) $$

Then label both train and test using the same fixed $\mathrm{thr}$:

$$ y_t = \mathbf{1}_{\{L_{t+1} > \mathrm{thr}\}} $$

### Interpretation
- The label definition is fixed by the *training* distribution.
- Test is evaluated “as-is” under the train-defined cutoff (no peeking).

### Pitfall
Computing $\mathrm{thr}$ using all data (train+test) leaks future distribution information and makes evaluation overly optimistic.

---

## 3) Time-based split (no shuffle)

### Definition
Use a chronological split:
- Train: $t \le T_{\text{split}}$
- Test: $t > T_{\text{split}}$

### Interpretation
This matches how the model would be used in practice: train on the past, evaluate on the future.

### Pitfall
Random shuffling breaks time ordering and can leak future patterns into training (a common mistake in time-series ML).

---

## 4) StandardScaler and pipelines

### Standardization (feature scaling)
StandardScaler computes mean and standard deviation on training features:

- Mean:

$$ \mu_j = \mathbb{E}[X_{t,j}] \quad \text{(train only)} $$

- Std:

$$ \sigma_j = \sqrt{\mathbb{V}[X_{t,j}]} \quad \text{(train only)} $$

Transforms each feature:

$$ z_{t,j} = \frac{x_{t,j} - \mu_j}{\sigma_j} $$

### Why it matters (Logistic Regression)
- Improves optimization stability
- Makes regularization behave fairly across features
- Coefficients become more comparable after scaling

### Pipeline best practice
Use a pipeline so the same train-fitted scaling is applied to test:
- fit scaler on train
- transform test with the same $(\mu,\sigma)$

### Pitfall
Fitting the scaler on the full dataset (train+test) is another form of leakage.

---

## 5) Rare-event evaluation metrics

### Confusion matrix counts
- TP: predicted 1 and true 1
- FP: predicted 1 but true 0
- FN: predicted 0 but true 1
- TN: predicted 0 and true 0

### Precision / recall
$$ \mathrm{precision} = \frac{TP}{TP+FP} $$
$$ \mathrm{recall} = \frac{TP}{TP+FN} $$

### Interpretation
- Precision: “When we raise an alert, how often is it truly extreme?”
- Recall: “Among true extreme days, how many did we catch?”

### Pitfall
When positives are rare, accuracy can look high even for a useless model (predicting 0 almost always).

---

## 6) ROC curve and ROC-AUC

### ROC curve definition
Vary a threshold $t$ on $\hat p_t$ and compute:

$$ \mathrm{TPR}(t) = \frac{TP(t)}{TP(t)+FN(t)} $$
$$ \mathrm{FPR}(t) = \frac{FP(t)}{FP(t)+TN(t)} $$

### ROC-AUC interpretation
ROC-AUC measures ranking quality of the score $\hat p_t$:

$$ \mathrm{AUC} \approx \mathbb{P}(s(X^{+}) > s(X^{-})) $$

### Pitfall
If test contains only one class, ROC-AUC is not defined.

---

## 7) Practical diagnostics (probabilities + threshold sweep)

### Probability diagnostics
Inspect the distribution of predicted probabilities:
- min/median/max
- quantiles
- mean $\hat p$ by class

### Red flags (common failure modes)
- predicted probabilities are tightly clustered (narrow range)
- mean $\hat p$ for $y=1$ is not higher than mean $\hat p$ for $y=0$
- threshold sweep flips from “almost none predicted positive” to “almost all predicted positive”

These indicate weak separation under the current feature/model setup.

---

## 8) Lagged return features (short-term dynamics)

### Motivation
Single-day return `ret_t` often has weak predictive signal for rare extreme events.  
Lagged returns summarize short-term dynamics (momentum / reversal / volatility bursts).

### Definition
Add lagged returns as features:

$$ \mathrm{ret\_lag}k(t) = r_{t-k} \quad (k=1,\dots,K) $$

Example feature vector (K=5):

$$ X_t = [r_t,\ \mathrm{vol20}_t,\ \mathrm{var250}_t,\ r_{t-1},\dots,r_{t-5}] $$

### Practical note
- `ret_lagk = ret.shift(k)` uses only information available at time $t$.
- Drop NaNs created by `diff`, `rolling`, and `shift` before training.

### Pitfall
Including “future lags” (e.g., `shift(-1)` inside features) leaks information.

---

## 9) Alert-budget evaluation (Recall@K)

### Why top-K alerts?
For rare events, fixed thresholds (e.g., 0.5) may be impractical.  
In real alerting, we often have a daily budget: “send only K alerts”.

### Definition (Recall@K)
Let $\hat p_t$ be a score and select the top-K dates by $\hat p_t$.

$$ \mathrm{Recall@K} = \frac{\#\{\text{true positives among top-K}\}}{\#\{\text{all true positives}\}} $$

### Interpretation
- High Recall@K means: “with only K alerts, we capture many extreme-loss days”.
- This is often more meaningful than accuracy for rare-event risk alerting.

### Pitfall
Recall@K can look unstable when the test period has very few positives; always report the positive count.

---

## 10) Linear vs non-linear baselines (Logistic vs RandomForest)

### Logistic Regression (linear score)
Logistic regression uses a linear score:

$$ s_t = \beta^\top X_t $$
$$ \hat p_t = \sigma(s_t) = \frac{1}{1+e^{-s_t}} $$

It may fail when the signal is non-linear or depends on interactions.

### Random Forest (non-linear baseline)
Random forests combine many decision trees and can capture non-linear rules and feature interactions  
(e.g., “high volatility AND negative lagged return”).

### Practical diagnostic
Compare models using:
- ROC-AUC (ranking ability)
- Alert-budget metrics (Recall@K)
- Threshold sweeps (precision/recall trade-off)

### Pitfall
Non-linear models can overfit; prefer time-based splits and keep hyperparameters conservative.