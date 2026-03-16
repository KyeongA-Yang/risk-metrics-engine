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

    # 6) Alert-budget evaluation (Recall@K)
    print("\n[Logit] === Alert-budget (Recall@K) ===")
    for k in [5, 10, 20, 50]:
        r_at_k = recall_at_k(y_test, proba_logit, k=k)
        print(f"Recall@{k}: {r_at_k:.4f}")

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

        print("\n[RF] === Alert-budget (Recall@K) ===")
        for k in [5, 10, 20, 50]:
            r_at_k = recall_at_k(y_test, proba_rf, k=k)
            print(f"Recall@{k}: {r_at_k:.4f}")


if __name__ == "__main__":
    main()
