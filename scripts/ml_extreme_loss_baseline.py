import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score


def main() -> None:
    # --- knobs (easy to tune) ---
    label_q = 0.90   # label threshold quantile (train-only)
    # threshold sweep will handle classification thresholds on predicted probability

    # 1) Load price data (SPY)
    df = pd.read_csv("data/price_SPY.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    price = df["price"].astype(float)

    # 2) Build returns / loss
    df_feat = pd.DataFrame(index=df.index)
    df_feat["ret"] = np.log(price).diff()
    df_feat["loss"] = -df_feat["ret"]

    # 3) Features (today's info)
    df_feat["vol20"] = df_feat["ret"].rolling(20).std()
    df_feat["var250"] = df_feat["ret"].rolling(250).quantile(0.01)  # 1% quantile of returns

    # 4) Label = whether tomorrow is extreme loss
    df_feat["loss_t1"] = df_feat["loss"].shift(-1)

    feat_cols = ["ret", "vol20", "var250"]

    # Drop rows with NaNs (from rolling, diff, shift)
    df_feat = df_feat.dropna(subset=feat_cols + ["loss_t1"]).copy()

    # 5) Time-based split (no shuffle)
    split = int(len(df_feat) * 0.8)
    train = df_feat.iloc[:split]
    test = df_feat.iloc[split:]

    # IMPORTANT: threshold must be computed on train only (avoid leakage)
    thr = train["loss_t1"].quantile(label_q)

    y_train = (train["loss_t1"] > thr).astype(int).to_numpy()
    y_test = (test["loss_t1"] > thr).astype(int).to_numpy()

    X_train = train[feat_cols].to_numpy()
    X_test = test[feat_cols].to_numpy()

    # 6) Model (Scaler + LogisticRegression)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs")
    )
    model.fit(X_train, y_train)

    # 7) Evaluate probabilities
    proba = model.predict_proba(X_test)[:, 1]

    # --- basic label diagnostics ---
    print(f"=== Label threshold on train (quantile={label_q}) ===")
    print("thr =", float(thr))
    print("train positives:", int(y_train.sum()), "out of", len(y_train), f"({y_train.mean():.4f})")
    print("test positives :", int(y_test.sum()), "out of", len(y_test), f"({y_test.mean():.4f})")

    # --- ROC-AUC (only if both classes exist in y_test) ---
    if len(np.unique(y_test)) == 2:
        print("ROC-AUC:", roc_auc_score(y_test, proba))
    else:
        print("ROC-AUC: not defined (only one class in y_test)")
    
    print("\n=== proba summary ===")
    print(pd.Series(proba).describe())

    print("\n=== proba quantiles ===")
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    print(pd.Series(proba).quantile(qs).to_string())

    proba_s = pd.Series(proba, index=test.index)
    y_s = pd.Series(y_test, index=test.index)

    print("\n=== mean proba by class ===")
    print("mean proba (y=0):", float(proba_s[y_s==0].mean()))
    print("mean proba (y=1):", float(proba_s[y_s==1].mean()))

    # --- Threshold sweep (rare-event friendly) ---
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05]

    print("\n=== Threshold sweep on test ===")
    print("thr  pred_pos  TP  FP  FN  TN  precision  recall")

    for t in thresholds:
        pred = (proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pred_pos = int(pred.sum())

        print(
            f"{t:>4.2f} {pred_pos:>8d} {tp:>3d} {fp:>3d} {fn:>3d} {tn:>3d} "
            f"{precision:>9.4f} {recall:>7.4f}"
        )


if __name__ == "__main__":
    main()