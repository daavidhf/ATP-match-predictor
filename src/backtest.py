"""Value-betting backtest: stake simulation, ROI, and drawdown.

This is the actual success criterion for the project (see README):
not "did the model predict the winner correctly" but "would betting
on the model's disagreements with the market have made money."
Everything here works on de-vigged probabilities (``src/odds.py``)
compared against a model's calibrated win probability, replayed
strictly in chronological order -- a backtest that shuffles match
order or peeks at future bankroll state isn't a backtest.

Expected input: a dataframe with one row per match, containing at
least:
  - a date column (default ``date``), sortable chronologically
  - ``p1_win_prob``: the model's calibrated P(P1 wins)
  - ``target``: 1 if P1 actually won, else 0
  - ``odds_p1``, ``odds_p2``: bookmaker decimal odds (ideally closing
    odds -- see README's note on why opening odds inflate the
    apparent edge)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import devig_normalize


def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Attach de-vigged ``market_p1``/``market_p2`` columns from the raw odds."""
    df = df.copy()
    implied_p1 = 1 / df["odds_p1"]
    implied_p2 = 1 / df["odds_p2"]
    total = implied_p1 + implied_p2
    df["market_p1"] = implied_p1 / total
    df["market_p2"] = implied_p2 / total
    return df


def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``edge_p1``/``edge_p2`` = model probability minus market probability.

    Because both the model's and the market's probabilities are
    binary (p2 = 1 - p1), ``edge_p1 == -edge_p2`` always: only one
    side can look like value on a given match.
    """
    df = df.copy()
    if "market_p1" not in df.columns:
        df = add_market_probabilities(df)
    df["edge_p1"] = df["p1_win_prob"] - df["market_p1"]
    df["edge_p2"] = (1 - df["p1_win_prob"]) - df["market_p2"]
    return df


def kelly_fraction(prob: float, odds: float, fraction: float = 1.0) -> float:
    """Fractional Kelly stake as a share of bankroll.

    Full Kelly ``f* = (b*p - q) / b`` where ``b = odds - 1`` and
    ``q = 1 - p``; ``fraction`` scales it down (e.g. 0.25 for
    "quarter Kelly"), the standard way to reduce variance from model
    and odds estimation error. Clipped to ``[0, 1]`` -- a negative
    Kelly fraction means "don't bet", not "bet against it" (this
    function only sizes a bet already selected as value).
    """
    b = odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    return float(np.clip(f_star * fraction, 0.0, 1.0))


def select_bets(df: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Pick, per match, which side (if any) clears the edge threshold.

    Adds ``bet_side`` (``"p1"``, ``"p2"``, or ``None``), ``bet_odds``,
    ``bet_prob`` (the model's probability for the chosen side), and
    ``bet_won`` (whether that side actually won).
    """
    df = compute_edges(df)

    bet_side = np.where(
        df["edge_p1"] > edge_threshold,
        "p1",
        np.where(df["edge_p2"] > edge_threshold, "p2", None),
    )
    df = df.copy()
    df["bet_side"] = bet_side
    df["bet_odds"] = np.where(df["bet_side"] == "p1", df["odds_p1"], df["odds_p2"])
    df["bet_prob"] = np.where(df["bet_side"] == "p1", df["p1_win_prob"], 1 - df["p1_win_prob"])
    df["bet_won"] = np.where(
        df["bet_side"] == "p1",
        df["target"] == 1,
        np.where(df["bet_side"] == "p2", df["target"] == 0, False),
    )
    return df


def simulate_backtest(
    df: pd.DataFrame,
    edge_threshold: float,
    stake_strategy: str = "flat",
    flat_stake: float = 1.0,
    kelly_fraction_mult: float = 0.25,
    initial_bankroll: float = 100.0,
    date_col: str = "date",
) -> dict:
    """Replay matches in chronological order, staking on every value bet found.

    Returns a dict with ``num_bets``, ``total_staked``, ``total_profit``,
    ``roi`` (profit / staked, ``None`` if no bets were placed),
    ``final_bankroll``, ``max_drawdown`` (largest peak-to-trough drop
    in the bankroll curve, as a fraction), and ``bets`` (the per-bet
    detail dataframe actually staked on, in the order they were
    played).
    """
    if stake_strategy not in ("flat", "kelly"):
        raise ValueError(f"Unknown stake_strategy: {stake_strategy!r}")

    scored = select_bets(df, edge_threshold).sort_values(date_col)
    bets = scored[scored["bet_side"].notna()].copy()

    bankroll = initial_bankroll
    bankroll_curve = [bankroll]
    stakes, profits = [], []

    for row in bets.itertuples(index=False):
        if stake_strategy == "flat":
            stake = flat_stake
        else:
            stake = bankroll * kelly_fraction(row.bet_prob, row.bet_odds, kelly_fraction_mult)

        profit = stake * (row.bet_odds - 1) if row.bet_won else -stake
        bankroll += profit
        stakes.append(stake)
        profits.append(profit)
        bankroll_curve.append(bankroll)

    bets["stake"] = stakes
    bets["profit"] = profits

    total_staked = float(sum(stakes))
    total_profit = float(sum(profits))
    roi = total_profit / total_staked if total_staked > 0 else None

    curve = pd.Series(bankroll_curve)
    running_peak = curve.cummax()
    drawdowns = (running_peak - curve) / running_peak.replace(0, np.nan)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

    return {
        "num_bets": len(bets),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "final_bankroll": bankroll,
        "max_drawdown": max_drawdown,
        "bets": bets,
    }


def choose_threshold_and_evaluate_holdout(
    df: pd.DataFrame,
    thresholds,
    final_season,
    season_col: str = "season_id",
    **backtest_kwargs,
) -> dict:
    """Select an edge threshold on validation seasons, then score it on the holdout alone.

    Mirrors the discipline ``train.season_splitter`` already applies
    to feature/hyperparameter decisions: picking the threshold that
    looks best by peeking at the same window you report results on
    turns the backtest into another leaky, self-fulfilling fixed
    holdout. Here, every season *before* ``final_season`` is fair game
    for picking the threshold (by best validation ROI among
    thresholds that placed at least one bet); ``final_season`` is
    untouched until that one threshold is evaluated on it.
    """
    validation_df = df[df[season_col] < final_season]
    holdout_df = df[df[season_col] == final_season]

    best_threshold, best_roi = None, float("-inf")
    for threshold in thresholds:
        result = simulate_backtest(validation_df, edge_threshold=threshold, **backtest_kwargs)
        if result["num_bets"] > 0 and result["roi"] is not None and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_threshold = threshold

    if best_threshold is None:
        return {"best_threshold": None, "validation_roi": None, "holdout_result": None}

    holdout_result = simulate_backtest(holdout_df, edge_threshold=best_threshold, **backtest_kwargs)
    return {
        "best_threshold": best_threshold,
        "validation_roi": best_roi,
        "holdout_result": holdout_result,
    }
