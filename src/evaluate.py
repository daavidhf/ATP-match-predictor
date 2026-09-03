"""Evaluation metrics beyond plain accuracy.

A predictor meant to output probabilities (not just a hard winner
pick) should be judged on calibration too -- accuracy alone can't
distinguish a well-calibrated 60% from an overconfident 90% that's
right just as often.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    """``y_proba``: predicted probability of the positive class (``target == 1``)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def calibration_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted_prob": prob_pred, "observed_freq": prob_true})


def baseline_higher_rank_wins(df: pd.DataFrame) -> np.ndarray:
    """Baseline: predict P1 wins iff P1 has the better (numerically lower) ATP rank.

    ``diff_rank`` is ``log(p1_rank) - log(p2_rank)``, so it's negative
    exactly when P1 is ranked higher. Rows with a missing rank default
    to predicting a P1 loss.
    """
    return (df["diff_rank"] < 0).astype(int).to_numpy()
