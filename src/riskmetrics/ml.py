from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def build_features(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Build leakage-safe features at time t (using info up to t).
    Label is based on loss at t+1 (next day).
    Returns a DataFrame with feature columns and 'loss_t1'.
    """
    df = df.copy()
    price = df["price"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["ret"] = np.log(price).diff()
    feat["loss"] = -feat["ret"]

    feat["vol20"] = feat["ret"].rolling(20).std()
    feat["var250"] = feat["ret"].rolling(250).quantile(0.01)

    for k in range(1, max_lag + 1):
        feat[f"ret_lag{k}"] = feat["ret"].shift(k)

    feat["loss_t1"] = feat["loss"].shift(-1)
    return feat


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total_pos = int(np.sum(y_true))
    if total_pos == 0:
        return float("nan")
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(np.sum(y_true[idx]))
    return tp / total_pos


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(np.sum(y_true[idx]))
    return tp / k


def walk_forward_expanding_indices(
    n: int,
    initial_train_frac: float = 0.6,
    n_splits: int = 5,
    min_test_size: int = 50,
):
    """
    Returns list of (train_idx, test_idx) with expanding train window.
    Indices are integer positions (0..n-1).
    """
    if not (0.1 < initial_train_frac < 0.95):
        raise ValueError("initial_train_frac must be in (0.1, 0.95)")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    init_train = int(n * initial_train_frac)
    if init_train < 100:
        raise ValueError("initial train too small")

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
    ks: list[int] = [10, 20, 50],
) -> pd.DataFrame:
    """
    Proper leakage-safe expanding walk-forward evaluation (Logit baseline).
    For each fold:
      - compute threshold on TRAIN only
      - label train/test using fold threshold
      - fit model on train, evaluate on test
    """
    X_all = feat[feat_cols].to_numpy()
    loss_t1_all = feat["loss_t1"].to_numpy()

    splits = walk_forward_expanding_indices(
        n=len(feat),
        initial_train_frac=initial_train_frac,
        n_splits=n_splits,
        min_test_size=50,
    )

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]

        thr_fold = float(np.quantile(loss_t1_all[tr_idx], label_q))
        y_tr = (loss_t1_all[tr_idx] > thr_fold).astype(int)
        y_te = (loss_t1_all[te_idx] > thr_fold).astype(int)

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
            "pos_rate_test": float(np.mean(y_te)),
            "auc": auc,
        }

        for k in ks:
            row[f"prec@{k}"] = precision_at_k(y_te, proba, k)
            row[f"rec@{k}"] = recall_at_k(y_te, proba, k)

        rows.append(row)

    return pd.DataFrame(rows)
