"""Build the neutral Player-A-vs-Player-B training dataset from raw ATP matches.

Each raw match row (``winner_*``/``loser_*`` columns) is turned into one
training row of ``p1_*``/``p2_*`` features with a randomized ``target``
(1 if P1 won, 0 otherwise), so the model can't learn "the winner column
always wins". All P1/P2 features are computed from state known strictly
*before* the match: pre-match Elo (see ``elo.py``) and rolling
historical stats (see ``features.py``).
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .elo import compute_elo
from .features import (
    LEVEL_MAP,
    ROUND_MAP,
    STATS_COLS,
    general_history,
    get_seed_value,
    h2h_history,
    is_lefty,
    update_general_history,
    update_h2h_history,
)


def _winner_player(row) -> dict:
    return {
        "id": row.winner_id,
        "elo_surf": row.w_elo_surf,
        "elo_gen": row.w_elo_gen,
        "atp_rank": row.winner_rank,
        "atp_points": row.winner_rank_points,
        "age": row.winner_age,
        "hand": row.winner_hand,
        "height": row.winner_ht,
        "ioc": row.winner_ioc,
        "seed": row.winner_seed,
    }


def _loser_player(row) -> dict:
    return {
        "id": row.loser_id,
        "elo_surf": row.l_elo_surf,
        "elo_gen": row.l_elo_gen,
        "atp_rank": row.loser_rank,
        "atp_points": row.loser_rank_points,
        "age": row.loser_age,
        "hand": row.loser_hand,
        "height": row.loser_ht,
        "ioc": row.loser_ioc,
        "seed": row.loser_seed,
    }


def _stat_row(row, prefix: str, won: int) -> dict:
    stats = {k: getattr(row, f"{prefix}_{k}") for k in STATS_COLS}
    stats["won"] = won
    return stats


def build_dataset(matches: pd.DataFrame, window_size: int = 10, seed: int | None = 42):
    """Replay ``matches`` chronologically and build the training dataframe.

    ``matches`` must already contain the raw Sackmann ATP columns
    (``winner_id``, ``loser_id``, ``surface``, ``tourney_date``, ...).
    Elo is computed internally via :func:`elo.compute_elo`.

    Returns ``(df_train, context)`` where ``context`` holds the *final*
    state after replaying every match (``elo_ratings``,
    ``history_general``, ``history_h2h``, ``last_match_minutes``) --
    exactly what :mod:`predict` needs to score a brand-new match.
    """
    matches = matches.sort_values("tourney_date").reset_index(drop=True)
    matches, elo_ratings = compute_elo(matches)

    rng = random.Random(seed)
    dataset = []
    history_general: dict = {}
    history_h2h: dict = {}
    last_match_minutes: dict = {}

    for row in matches.itertuples(index=False):
        winner = _winner_player(row)
        loser = _loser_player(row)

        if rng.random() > 0.5:
            p1, p2, target = winner, loser, 1
        else:
            p1, p2, target = loser, winner, 0

        p1_lefty = is_lefty(p1)
        p2_lefty = is_lefty(p2)
        is_same_country = 1 if p1["ioc"] == p2["ioc"] else 0
        round_value = ROUND_MAP.get(row.round)
        tl_value = LEVEL_MAP.get(row.tourney_level)

        p1_stats = general_history(history_general, p1)
        p2_stats = general_history(history_general, p2)
        p1_h2h_stats, p2_h2h_stats, matches_num, p1_h2h_win_pct = h2h_history(history_h2h, p1, p2)

        p1_last_mins = last_match_minutes.get(p1["id"], 0)
        p2_last_mins = last_match_minutes.get(p2["id"], 0)

        p1_seed = get_seed_value(p1)
        p2_seed = get_seed_value(p2)

        dataset.append(
            {
                "date": row.tourney_date,
                "diff_elo_surf": p1["elo_surf"] - p2["elo_surf"],
                "diff_elo_gen": p1["elo_gen"] - p2["elo_gen"],
                "diff_rank": np.log(p1["atp_rank"]) - np.log(p2["atp_rank"]),
                "diff_points": p1["atp_points"] - p2["atp_points"],
                "diff_age": p1["age"] - p2["age"],
                "diff_height": p1["height"] - p2["height"],
                "diff_last_minutes": p1_last_mins - p2_last_mins,
                "same_country": is_same_country,
                "surface": row.surface,
                "tourney_level": tl_value,
                "best_of": row.best_of,
                "draw_size": row.draw_size,
                "round": round_value,
                "diff_seed": p1_seed - p2_seed,
                "p1_is_lefty": p1_lefty,
                "p1_recent_win_pct": p1_stats.get("recent_win_pct"),
                "p1_ace": p1_stats.get("avg_ace"),
                "p1_df": p1_stats.get("avg_df"),
                "p1_1stWon": p1_stats.get("pct_1stWon"),
                "p1_bpSaved": p1_stats.get("avg_bpSaved"),
                "p1_pct_1sIn": p1_stats.get("pct_1stIn"),
                "p1_pct_1sWon": p1_stats.get("pct_1stWon"),
                "p1_pct_2ndWon": p1_stats.get("pct_2ndWon"),
                "p2_is_lefty": p2_lefty,
                "p2_recent_win_pct": p2_stats.get("recent_win_pct"),
                "p2_ace": p2_stats.get("avg_ace"),
                "p2_df": p2_stats.get("avg_df"),
                "p2_1stWon": p2_stats.get("pct_1stWon"),
                "p2_bpSaved": p2_stats.get("avg_bpSaved"),
                "p2_pct_1sIn": p2_stats.get("pct_1stIn"),
                "p2_pct_1sWon": p2_stats.get("pct_1stWon"),
                "p2_pct_2ndWon": p2_stats.get("pct_2ndWon"),
                "h2h_matches": matches_num,
                "h2h_p1_win_pct": p1_h2h_win_pct,
                "p1_h2h_ace": p1_h2h_stats.get("avg_ace"),
                "p1_h2h_df": p1_h2h_stats.get("avg_df"),
                "p1_h2h_1stWon": p1_h2h_stats.get("pct_1stWon"),
                "p1_h2h_bpSaved": p1_h2h_stats.get("avg_bpSaved"),
                "p1_h2h_pct_1sIn": p1_h2h_stats.get("pct_1stIn"),
                "p1_h2h_pct_1sWon": p1_h2h_stats.get("pct_1stWon"),
                "p1_h2h_pct_2ndWon": p1_h2h_stats.get("pct_2ndWon"),
                "p2_h2h_ace": p2_h2h_stats.get("avg_ace"),
                "p2_h2h_df": p2_h2h_stats.get("avg_df"),
                "p2_h2h_1stWon": p2_h2h_stats.get("pct_1stWon"),
                "p2_h2h_bpSaved": p2_h2h_stats.get("avg_bpSaved"),
                "p2_h2h_pct_1sIn": p2_h2h_stats.get("pct_1stIn"),
                "p2_h2h_pct_1sWon": p2_h2h_stats.get("pct_1stWon"),
                "p2_h2h_pct_2ndWon": p2_h2h_stats.get("pct_2ndWon"),
                "target": target,
            }
        )

        if row.winner_id == p1["id"]:
            stats_p1_now = _stat_row(row, "w", won=1)
            stats_p2_now = _stat_row(row, "l", won=0)
        else:
            stats_p1_now = _stat_row(row, "l", won=1)
            stats_p2_now = _stat_row(row, "w", won=0)

        update_general_history(history_general, p1, stats_p1_now, max_len=window_size)
        update_general_history(history_general, p2, stats_p2_now, max_len=window_size)
        update_h2h_history(history_h2h, p1, p2, stats_p1_now, stats_p2_now)

        match_minutes = row.minutes if pd.notna(row.minutes) else 0
        last_match_minutes[p1["id"]] = match_minutes
        last_match_minutes[p2["id"]] = match_minutes

    df_train = pd.DataFrame(dataset)
    df_train["p1_is_lefty"] = df_train["p1_is_lefty"].astype("Int64")
    df_train["p2_is_lefty"] = df_train["p2_is_lefty"].astype("Int64")
    df_train = pd.get_dummies(df_train, columns=["surface"], drop_first=True)
    df_train = df_train.sort_values("date").reset_index(drop=True)

    context = {
        "elo_ratings": elo_ratings,
        "history_general": history_general,
        "history_h2h": history_h2h,
        "last_match_minutes": last_match_minutes,
    }
    return df_train, context
