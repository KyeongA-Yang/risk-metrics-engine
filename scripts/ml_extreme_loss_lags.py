from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

# Optional non-linear baseline
from sklearn.ensemble import RandomForestClassifier


def build_features(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Build leakage-safe features at time t (using info up to t).
    Label is based on loss at t+1 (next day).

    Returns a DataFrame with features and label column loss_t1.
    """
    df = df.copy()

    price = df["price"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["ret"] = np.log(price).diff()
    feat["loss"] = -feat["ret"]

    # volatility / var-like state features (computed using past returns up to t)
    feat["vol20"] = feat["ret"].rolling(20).std()
    feat["var250"] = feat["ret"].rolling(250).quantile(0.01)

    # lagged returns (past info only)
    for k in range(1, max_lag + 1):
        feat[f"ret_lag{k}"] = feat["ret"].shift(k)

    # label = tomorrow's loss
    feat["loss_t1"] = feat["loss"].shift(-1)

    return feat


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Alert-budget evaluation: among top-k scores, how many true positives captured?
    recall@k = TP_in_topk / total_positives
    """
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return float("nan")

    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(y_true[idx].sum())
    return tp / total_pos


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Alert-budget evaluation: among top-k scores, what fraction are true positives?
    precision@k = TP_in_topk / k
    """
    if len(scores) == 0:
        return float("nan")

    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(y_true[idx].sum())
    return tp / k


def walk_forward_expanding_indices(
    n: int,
    initial_train_frac: float = 0.6,
    n_splits: int = 5,
    min_test_size: int = 50,
):
    if not (0.1 < initial_train_frac < 0.95):
        raise ValueError("initial_train_frac must be in (0.1, 0.95)")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    init_train = int(n * initial_train_frac)
    if init_train < 100:
        raise ValueError("initial train too small; increase initial_train_frac or use more data")

    remaining = n - init_train
    test_block = max(min_test_size, remaining // n_splits)

    splits = []
    for i in range(n_splits):
        train_end = init_train + i * test_block
        test_start = train_end
        test_end = min(test_start + test_block, n)

        if test_start >= n:
            break
        if test_end - test_start < min_test_size:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def eval_walk_forward_expanding(
    feat: pd.DataFrame,
    feat_cols: list[str],
    label_q: float = 0.90,
    initial_train_frac: float = 0.6,
    n_splits: int = 5,
    ks: list[int] = [5, 10, 20, 50],
):
    """
    Proper leakage-safe walk-forward:
    - For each fold:
      1) compute thr on train only
      2) build y_train/y_test using fold-specific thr
      3) fit model on train, evaluate on test
    """
    X_all = feat[feat_cols].to_numpy()
    loss_t1_all = feat["loss_t1"].to_numpy()  # continuous target (tomorrow loss)

    splits = walk_forward_expanding_indices(
        n=len(feat),
        initial_train_frac=initial_train_frac,
        n_splits=n_splits,
        min_test_size=50,
    )

    rows = []

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]

        # --- fold-specific threshold computed on TRAIN ONLY ---
        thr_fold = float(np.quantile(loss_t1_all[tr_idx], label_q))

        y_tr = (loss_t1_all[tr_idx] > thr_fold).astype(int)
        y_te = (loss_t1_all[te_idx] > thr_fold).astype(int)

        # --- fit LOGIT each fold ---
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced"),
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) == 2 else float("nan")

        row = {
            "fold": fold,
            "thr_fold": thr_fold,
            "n_test": len(y_te),
            "pos_test": int(y_te.sum()),
            "pos_rate_test": float(y_te.mean()),
            "auc": auc,
        }

        # alert-budget metrics
        for k in ks:
            row[f"prec@{k}"] = precision_at_k(y_te, proba, k)
            row[f"rec@{k}"] = recall_at_k(y_te, proba, k)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def main() -> None:
    # --- knobs ---
    label_q = 0.90   # top 10% tomorrow-loss days as "extreme" (train-only threshold)
    max_lag = 5
    use_random_forest = True  # optional comparison

    # 1) Load SPY
    df = pd.read_csv("data/price_SPY.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # 2) Build features + label
    feat = build_features(df, max_lag=max_lag)

    # Choose feature columns
    feat_cols = ["ret", "vol20", "var250"] + [f"ret_lag{k}" for k in range(1, max_lag + 1)]

    # Drop rows with NaNs (diff/rolling/lag/shift)
    feat = feat.dropna(subset=feat_cols + ["loss_t1"]).copy()

    # 3) Time split 80/20
    split = int(len(feat) * 0.8)
    train = feat.iloc[:split]
    test = feat.iloc[split:]

    # 4) Train-only label threshold (avoid leakage)
    thr = train["loss_t1"].quantile(label_q)
    y_train = (train["loss_t1"] > thr).astype(int).to_numpy()
    y_test = (test["loss_t1"] > thr).astype(int).to_numpy()

    X_train = train[feat_cols].to_numpy()
    X_test = test[feat_cols].to_numpy()

    print(f"=== Label threshold on train (quantile={label_q}) ===")
    print("thr =", float(thr))
    print("train positives:", int(y_train.sum()), "out of", len(y_train), f"({y_train.mean():.4f})")
    print("test positives :", int(y_test.sum()), "out of", len(y_test), f"({y_test.mean():.4f})")

    # 5) Logistic baseline (scaled)
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced"),
    )
    logit.fit(X_train, y_train)
    proba_logit = logit.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, proba_logit)
        print("\n[Logit] ROC-AUC:", auc)
    else:
        print("\n[Logit] ROC-AUC: not defined (only one class in y_test)")

    # Threshold sweep (rare-event friendly)
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05]
    print("\n[Logit] === Threshold sweep on test ===")
    print("thr  pred_pos  TP  FP  FN  TN  precision  recall")

    for t in thresholds:
        pred = (proba_logit >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pred_pos = int(pred.sum())
        print(f"{t:>4.2f} {pred_pos:>8d} {tp:>3d} {fp:>3d} {fn:>3d} {tn:>3d} {precision:>9.4f} {recall:>7.4f}")

    # 6) Alert-budget evaluation (Precision@K + Recall@K) — LOGIT
    print("\n[Logit] === Alert-budget (Precision@K / Recall@K) ===")
    for k in [5, 10, 20, 50]:
        p_at_k = precision_at_k(y_test, proba_logit, k=k)
        r_at_k = recall_at_k(y_test, proba_logit, k=k)
        print(f"K={k:>2d}  Precision@K: {p_at_k:.4f}  Recall@K: {r_at_k:.4f}")

    # 6.5) Interpretation: Logistic coefficients (after scaling)
    lr = logit.named_steps["logisticregression"]
    coefs = lr.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feat_cols, "coef": coefs}).sort_values("coef", ascending=False)

    print("\n[Logit] === Coefficients (standardized features) ===")
    print(coef_df.to_string(index=False))

    print("\n[Logit] Top positive signals (increase P(y=1)):")
    print(coef_df.head(5).to_string(index=False))

    print("\n[Logit] Top negative signals (decrease P(y=1)):")
    print(coef_df.tail(5).to_string(index=False))

    # 7) Optional: RandomForest comparison (non-linear baseline)
    if use_random_forest:
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=0,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
        )
        rf.fit(X_train, y_train)
        proba_rf = rf.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) == 2:
            auc_rf = roc_auc_score(y_test, proba_rf)
            print("\n[RF] ROC-AUC:", auc_rf)
        else:
            print("\n[RF] ROC-AUC: not defined (only one class in y_test)")

        # RF alert-budget
        print("\n[RF] === Alert-budget (Precision@K / Recall@K) ===")
        for k in [5, 10, 20, 50]:
            p_at_k = precision_at_k(y_test, proba_rf, k=k)
            r_at_k = recall_at_k(y_test, proba_rf, k=k)
            print(f"K={k:>2d}  Precision@K: {p_at_k:.4f}  Recall@K: {r_at_k:.4f}")

        # RF feature importance
        imp = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values("importance", ascending=False)

        print("\n[RF] === Feature importance ===")
        print(imp_df.to_string(index=False))

        print("\n[RF] Top features:")
        print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()