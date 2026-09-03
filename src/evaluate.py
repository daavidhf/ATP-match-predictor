"""Evaluation metrics for a *probability* predictor, not just a classifier.

The project's actual goal is a well-calibrated win probability to
compare against the market's de-vigged probability (src/odds.py) and
backtest as bets (src/backtest.py) -- not maximizing accuracy. A model
can be more accurate and worse for that purpose (e.g. overconfident:
right as often, but wrong about *how* likely it was), so log-loss,
Brier score, and calibration are the metrics that actually matter
here; accuracy is kept only as a cheap sanity check that the model
learned real signal at all, not as the number to optimize.

``baseline_higher_rank_wins`` below is that same kind of sanity check,
not the real bar either: the bar that matters is whether the model's
probability disagrees with the market's own (de-vigged, ideally
closing) odds -- see src/backtest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    """``y_proba``: predicted probability of the positive class (``target == 1``).

    Ordered by what actually matters for this project: log-loss and
    Brier score (calibration/probability quality) first, accuracy last
    as a sanity check.
    """
    return {
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def calibration_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: within each predicted-probability bucket, how often did P1 actually win?

    A well-calibrated model has ``observed_freq`` close to
    ``predicted_prob`` in every row -- systematic gaps are exactly
    where a de-vigged market comparison could turn into a real edge
    (or reveal the model is the one that's wrong).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted_prob": prob_pred, "observed_freq": prob_true})


def baseline_higher_rank_wins(df: pd.DataFrame) -> np.ndarray:
    """Sanity-check baseline: predict P1 wins iff P1 has the better (numerically lower) ATP rank.

    ``diff_rank`` is ``log(p1_rank) - log(p2_rank)``, so it's negative
    exactly when P1 is ranked higher. Rows with a missing rank default
    to predicting a P1 loss. Useful to confirm the model beats a
    trivial rule, but the number that decides whether this project
    "works" is the backtest ROI against market odds, not this.
    """
    return (df["diff_rank"] < 0).astype(int).to_numpy()
