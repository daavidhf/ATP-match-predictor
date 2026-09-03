"""Regression tests for audit bug #2: the ``is not np.nan`` identity check.

A NaN read out of a pandas cell (e.g. via ``.iloc``) is never the same
*object* as the literal ``np.nan`` -- ``pd.Series([1.0, np.nan]).iloc[1]
is not np.nan`` is ``True``, even though the value *is* NaN. The
original filter therefore let NaNs straight through into the average,
which silently corrupted ~65% of the detailed-stats rows across the
dataset. ``get_avg_stats`` must use ``pd.notna`` instead.
"""
import numpy as np
import pandas as pd

from src.features import get_avg_stats, get_days_rest, get_seed_value, is_lefty


def _pandas_nan():
    """A NaN produced by pandas, not the ``np.nan`` singleton -- reproduces
    the identity mismatch that made the original filter a no-op."""
    return pd.Series([1.0, np.nan]).iloc[1]


def test_pandas_nan_is_not_the_np_nan_object():
    # Documents *why* the bug existed: this is the surprising behavior
    # the original `match[k] is not np.nan` check relied on being False.
    val = _pandas_nan()
    assert val is not np.nan
    assert pd.isna(val)


def test_get_avg_stats_excludes_nan_values():
    history = [
        {"ace": 4.0, "won": 1},
        {"ace": _pandas_nan(), "won": 0},
        {"ace": 6.0, "won": 1},
    ]
    avgs = get_avg_stats(history)
    # Average of 4 and 6 only -- the NaN row must be dropped, not
    # treated as a real value (which would make it NaN, or, with the
    # old identity check, still slip through unfiltered).
    assert avgs["avg_ace"] == 5.0


def test_get_avg_stats_all_nan_column_defaults_to_zero():
    history = [{"ace": _pandas_nan(), "won": 0}, {"ace": _pandas_nan(), "won": 1}]
    avgs = get_avg_stats(history)
    assert avgs["avg_ace"] == 0


def test_get_avg_stats_empty_history_returns_empty_dict():
    assert get_avg_stats([]) == {}


def test_recent_win_pct_matches_win_fraction():
    history = [{"won": 1}, {"won": 0}, {"won": 1}, {"won": 1}]
    avgs = get_avg_stats(history)
    assert avgs["recent_win_pct"] == 0.75


def test_is_lefty():
    assert is_lefty({"hand": "L"}) == 1.0
    assert is_lefty({"hand": "R"}) == 0.0
    assert np.isnan(is_lefty({"hand": "U"}))


def test_get_seed_value_defaults_when_missing():
    assert get_seed_value({"seed": np.nan}) == 36
    assert get_seed_value({"seed": 4}) == 4


def test_get_days_rest_clamps_and_defaults():
    last_match_date = {1: pd.Timestamp("2022-01-01")}
    # First-ever match for a player not in the dict: neutral default.
    assert get_days_rest(last_match_date, {"id": 2}, pd.Timestamp("2022-01-10")) == 30
    # Normal case.
    assert get_days_rest(last_match_date, {"id": 1}, pd.Timestamp("2022-01-11")) == 10
    # Clamped above 60.
    assert get_days_rest(last_match_date, {"id": 1}, pd.Timestamp("2022-06-01")) == 60
