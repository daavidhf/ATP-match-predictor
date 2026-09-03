"""Model training and validation splits."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEFAULT_MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
)


def make_model(**overrides) -> RandomForestClassifier:
    params = {**DEFAULT_MODEL_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def time_split(df_train: pd.DataFrame, cutting_date):
    """Split chronologically: everything before ``cutting_date`` is train, the rest test.

    A random ``train_test_split`` would let the model train on matches
    that happened after the ones it's tested on -- this is time-series
    data, so the split must respect chronology.
    """
    cutting_date = pd.to_datetime(cutting_date)
    train_mask = df_train["date"] < cutting_date
    test_mask = df_train["date"] >= cutting_date

    X_train = df_train.loc[train_mask].drop(columns=["target", "date"])
    y_train = df_train.loc[train_mask, "target"]
    X_test = df_train.loc[test_mask].drop(columns=["target", "date"])
    y_test = df_train.loc[test_mask, "target"]
    return X_train, y_train, X_test, y_test


def season_id(date_series: pd.Series) -> pd.Series:
    """Tennis-season id: a December match belongs to next year's season."""
    return date_series.dt.year + (date_series.dt.month >= 12).astype(int)


def season_splitter(df: pd.DataFrame, season_col: str):
    """Yield expanding-window ``(train_idx, test_idx)`` pairs, one per season.

    Training accumulates every past season; test is always the single
    season immediately following. This is the walk-forward validation
    that should drive feature/hyperparameter decisions -- unlike a
    single fixed holdout, it can't be silently overfit by repeated
    peeking.
    """
    seasons = sorted(df[season_col].unique())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 complete seasons to validate.")

    for i in range(1, len(seasons)):
        past_seasons = seasons[:i]
        current_season = seasons[i]
        train_idx = df[df[season_col].isin(past_seasons)].index.values
        test_idx = df[df[season_col] == current_season].index.values
        yield train_idx, test_idx


def walk_forward_splits(df_train: pd.DataFrame):
    """Convenience wrapper: attach ``season_id`` and return ``(X, y, splits)``."""
    df = df_train.copy()
    df["season_id"] = season_id(df["date"])
    X = df.drop(columns=["target", "date", "season_id"])
    y = df["target"]
    splits = list(season_splitter(df, "season_id"))
    return X, y, splits
