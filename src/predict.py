"""Inference on hypothetical, not-yet-played matches.

Loads the players/rankings reference tables once (via ``load_players``
/ ``load_rankings``) instead of re-reading the full CSVs from disk on
every single lookup, and looks up a player's Elo through
``EloRatings.get_surface`` instead of indexing the per-surface dict
directly by player id (audit bug #4: ``elo_surface`` is keyed by
surface first, ``{'Hard': {...}, 'Clay': {...}, ...}``, so
``elo_surface.get(player_id, 1500)`` always missed and silently fell
back to the default, regardless of the player or the match surface).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .elo import EloRatings
from .features import (
    LEVEL_MAP,
    ROUND_MAP,
    general_history,
    get_seed_value,
    h2h_history,
    is_lefty,
)


def load_players(path: str) -> pd.DataFrame:
    players = pd.read_csv(path)
    players["dob"] = pd.to_datetime(players["dob"], format="%Y%m%d", errors="coerce")
    return players.set_index("player_id", drop=False)


def load_rankings(path: str) -> pd.DataFrame:
    rankings = pd.read_csv(path)
    rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d", errors="coerce")
    return rankings


def get_player_info(
    player_id,
    players: pd.DataFrame,
    rankings: pd.DataFrame,
    elo_ratings: EloRatings,
    surface: str,
    date=None,
) -> dict:
    """Snapshot of a player's known state as of ``date`` (defaults to today)."""
    if date is None:
        date = pd.Timestamp(datetime.now())
    else:
        date = pd.to_datetime(date, format="%Y%m%d", errors="coerce")

    if player_id not in players.index:
        raise ValueError(f"Player ID {player_id} not found in players dataset.")
    player_dict = players.loc[player_id].to_dict()

    player_ranks = rankings[rankings["player"] == player_id]
    past_ranks = player_ranks[player_ranks["ranking_date"] <= date]
    if past_ranks.empty:
        raise ValueError(f"No ATP ranking found for player ID {player_id} before date {date}.")
    latest_rank = past_ranks.sort_values("ranking_date", ascending=False).iloc[0]

    dob = player_dict["dob"]
    age = date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))

    return {
        "id": player_id,
        "elo_surf": elo_ratings.get_surface(player_id, surface),
        "elo_gen": elo_ratings.get_general(player_id),
        "atp_rank": latest_rank["rank"],
        "atp_points": latest_rank["points"],
        "age": age,
        "hand": is_lefty(player_dict),
        "height": player_dict["height"],
        "ioc": player_dict["ioc"],
    }


def predict_future_match(
    tourney: dict,
    p1: dict,
    p2: dict,
    history_general: dict,
    history_h2h: dict,
    last_match_minutes: dict,
) -> dict:
    """Build the same feature vector used in training for a hypothetical match."""
    p1_lefty = is_lefty(p1)
    p2_lefty = is_lefty(p2)
    is_same_country = 1 if p1["ioc"] == p2["ioc"] else 0
    round_value = ROUND_MAP.get(tourney["round"])
    tl_value = LEVEL_MAP.get(tourney["tourney_level"])

    p1_stats = general_history(history_general, p1)
    p2_stats = general_history(history_general, p2)
    p1_h2h_stats, p2_h2h_stats, matches_num, p1_h2h_win_pct = h2h_history(history_h2h, p1, p2)

    p1_last_mins = last_match_minutes.get(p1["id"], 0)
    p2_last_mins = last_match_minutes.get(p2["id"], 0)

    p1_seed = get_seed_value(p1)
    p2_seed = get_seed_value(p2)

    surface = tourney["surface"]

    return {
        "diff_elo_surf": p1["elo_surf"] - p2["elo_surf"],
        "diff_elo_gen": p1["elo_gen"] - p2["elo_gen"],
        "diff_rank": np.log(p1["atp_rank"]) - np.log(p2["atp_rank"]),
        "diff_points": p1["atp_points"] - p2["atp_points"],
        "diff_age": p1["age"] - p2["age"],
        "diff_height": p1["height"] - p2["height"],
        "diff_last_minutes": p1_last_mins - p2_last_mins,
        "same_country": is_same_country,
        "tourney_level": tl_value,
        "best_of": tourney["best_of"],
        "draw_size": tourney["draw_size"],
        "round": round_value,
        "diff_seed": p1_seed - p2_seed,
        "surface_Clay": 1 if surface == "Clay" else 0,
        "surface_Grass": 1 if surface == "Grass" else 0,
        "surface_Hard": 1 if surface == "Hard" else 0,
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
    }


def get_real_probabilities(odd_p1: float, odd_p2: float):
    """Normalize betting odds (which include the bookmaker's margin) into probabilities."""
    implied_p1 = 1 / odd_p1
    implied_p2 = 1 / odd_p2
    total_implied = implied_p1 + implied_p2
    return implied_p1 / total_implied * 100, implied_p2 / total_implied * 100
