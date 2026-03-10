# ML Concepts
> Definitions, formulas, intuition, and pitfalls for ML tasks.

This note defines core ML concepts used for time-series classification in this repo.
**Convention used throughout:**
- We build features using information available at time $t$
- We predict an outcome at time $t+1$ (next-day event)
- All preprocessing parameters (e.g., scaling) are learned on train only

---

## 0) Notation
- Price: $P_t$
- Log return: $ r_t = \log(P_t) - \log(P_{t-1}) $
- Loss: $ L_t = -r_t $
- Feature vector at time $t$: $X_t$
- Binary label at time $t$: $y_t \in \{0,1\}$
- Train/test split index: $T_{\text{split}}$
- Predicted probability (score): $\hat p_t = \mathbb{P}(y_t=1 \mid X_t)$
- Quantile operator: $q_\alpha(\cdot)$

---

## 1) Problem setup (next-day extreme-loss classification)
### Definition
We predict whether tomorrow is an extreme-loss day:


$$ y_t = \mathbf{1}_{\{L_{t+1} > \text{thr}\}} $$


### Interpretation
- $y_t=1$ means: “tomorrow ($t+1$) loss is unusually large”
- Features $X_t$ must use only information up to time $t$

### Practical note
A common implementation is to create next-day loss via shifting:
- `loss_t1 = loss.shift(-1)` so `loss_t1(t) = loss(t+1)`

---

## 2) Leakage and leakage-safe labeling
### Leakage-free thresholding
The extreme-loss cutoff must be computed using the training period only:

$$ \text{thr} = q_{\text{label\_q}}\left(\{L_{t+1}: t \in \text{train}\}\right) $$


Then label both train and test using the same $\text{thr}$:
$ y_t = \mathbf{1}_{\{L_{t+1} > \text{thr}\}} $

### Pitfall
Computing $\text{thr}$ using all data (train+test) leaks future distribution information into training and makes evaluation overly optimistic.

---

## 3) Time-based split (no shuffle)
### Definition
Use chronological split:
- Train: $t \le T_{\text{split}}$
- Test: $t > T_{\text{split}}$

### Pitfall
Random shuffling breaks time ordering and can leak future patterns into training (common mistake in time-series ML).

---

## 4) StandardScaler and pipelines
### Standardization (feature scaling)
StandardScaler computes mean and standard deviation on training features:
- Mean: $ \mu_j = \mathbb{E}[X_{t,j}] \ \text{(train only)} $
- Std: $ \sigma_j = \sqrt{\mathbb{V}[X_{t,j}]} \ \text{(train only)} $

Transforms each feature:

$$ z_{t,j} = \frac{x_{t,j} - \mu_j}{\sigma_j} $$


### Why it matters (Logistic Regression)
- Improves optimization stability
- Makes regularization behave fairly across features
- Coefficients become more comparable

### Pipeline best practice
Use a pipeline so the same train-fitted scaling is applied to test:
- fit scaler on train
- transform test with the same $(\mu,\sigma)$

---

## 5) Rare-event evaluation metrics
### Confusion matrix counts
- TP: predicted 1 and true 1
- FP: predicted 1 but true 0
- FN: predicted 0 but true 1
- TN: predicted 0 and true 0

### Precision / recall
$$ \text{precision} = \frac{TP}{TP+FP} $$
$$ \text{recall} = \frac{TP}{TP+FN} $$

### Practical note
When positives are rare, accuracy is often misleading. Precision/recall (and threshold sweeps) are more informative.

---

## 6) ROC curve and ROC-AUC
### ROC curve definition
Vary a threshold $t$ on $\hat p_t$ and compute:

$$ \text{TPR}(t) = \frac{TP(t)}{TP(t)+FN(t)} $$

$$ \text{FPR}(t) = \frac{FP(t)}{FP(t)+TN(t)} $$


### ROC-AUC interpretation
ROC-AUC measures ranking quality of the score $\hat p_t$:
$ \text{AUC} \approx \mathbb{P}(s(X^+) > s(X^-)) $

### Pitfall
If test contains only one class, ROC-AUC is not defined.

---

## 7) Practical diagnostics (probabilities + threshold sweep)
### Probability diagnostics
Inspect the distribution of predicted probabilities:
- min/median/max
- quantiles
- mean $\hat p$ by class

### Red flags (common failure mode)
- predicted probabilities are tightly clustered (narrow range)
- mean $\hat p$ for $y=1$ is not higher than mean $\hat p$ for $y=0$
- threshold sweep flips from “almost none predicted positive” to “almost all predicted positive”

These indicate weak separation under the current feature/model setup.