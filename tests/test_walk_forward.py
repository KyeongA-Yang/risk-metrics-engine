import numpy as np
import pandas as pd

from riskmetrics.ml import (
    walk_forward_expanding_indices,
    eval_walk_forward_expanding,
    build_features,
)


def test_walk_forward_expanding_indices_expands_train_and_moves_forward():
    n = 1000
    splits = walk_forward_expanding_indices(
        n=n, initial_train_frac=0.6, n_splits=5, min_test_size=50
    )

    # should produce multiple folds
    assert len(splits) >= 2

    prev_train_len = 0
    prev_test_end = -1

    for tr_idx, te_idx in splits:
        # non-empty blocks
        assert len(tr_idx) > 0
        assert len(te_idx) > 0

        # expanding: train grows over folds
        assert len(tr_idx) > prev_train_len
        prev_train_len = len(tr_idx)

        # no overlap: test starts after train ends
        assert tr_idx[-1] < te_idx[0]

        # time moves forward (test blocks do not go backwards)
        assert te_idx[0] > prev_test_end
        prev_test_end = te_idx[-1]


def test_eval_walk_forward_expanding_returns_expected_columns_and_bounds():
    # make synthetic price series (so build_features works)
    n = 700
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    # lognormal-ish price path
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    df = pd.DataFrame({"price": price}, index=dates)

    feat = build_features(df, max_lag=3)

    feat_cols = ["ret", "vol20", "var250"] + [f"ret_lag{k}" for k in range(1, 4)]
    feat = feat.dropna(subset=feat_cols + ["loss_t1"]).copy()

    out = eval_walk_forward_expanding(
        feat=feat,
        feat_cols=feat_cols,
        label_q=0.90,
        initial_train_frac=0.6,
        n_splits=3,
        ks=[10, 20],
    )

    # at least one fold
    assert len(out) >= 1

    required_cols = {
        "fold",
        "thr_fold",
        "n_test",
        "pos_test",
        "pos_rate_test",
        "auc",
        "prec@10",
        "rec@10",
        "prec@20",
        "rec@20",
    }
    assert required_cols.issubset(set(out.columns))

    # folds start at 1
    assert int(out["fold"].min()) == 1

    # sizes are positive
    assert (out["n_test"] > 0).all()

    # rates in [0,1]
    assert ((out["pos_rate_test"] >= 0) & (out["pos_rate_test"] <= 1)).all()
    assert ((out["prec@10"] >= 0) & (out["prec@10"] <= 1)).all()
    assert ((out["rec@10"] >= 0) & (out["rec@10"] <= 1)).all()

    # thresholds should be finite numbers
    assert np.isfinite(out["thr_fold"]).all()
