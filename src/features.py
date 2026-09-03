"""Categorical mappings and player-history feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROUND_MAP = {
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "BR": 6.5,  # Bronze Medal (Third-place play-off between SF and Final)
    "F": 7,
    "RR": 8,  # Round Robin (usually the ATP Finals, a very high level)
    "ER": 1,  # Early Round (Davis Cup or very low rounds, minimum value)
}

LEVEL_MAP = {
    "G": 5,  # Grand Slam
    "F": 4,  # Tour Finals
    "M": 3,  # Masters 1000
    "A": 2,  # ATP 250/500
    "D": 2,  # Davis Cup
    "C": 1,  # Challengers
    "S": 1,  # Satellites/Futures
}

STATS_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]

DEFAULT_SEED = 36


def is_lefty(player) -> float:
    """1.0 if left-handed (``L``), 0.0 if right-handed (``R``), NaN if unknown."""
    hand = player["hand"]
    if hand == "L":
        return 1.0
    if hand == "R":
        return 0.0
    return np.nan


def get_seed_value(player, default: float = DEFAULT_SEED) -> float:
    seed = player["seed"]
    if pd.isnull(seed):
        return default
    return seed


def get_avg_stats(history_list: list[dict]) -> dict:
    """Average the per-match stats recorded in ``history_list``.

    Missing values are dropped with ``pd.notna`` rather than ``is not
    np.nan``. A NaN read out of a pandas cell is never the same
    *object* as the literal ``np.nan``, so ``is not np.nan`` was true
    even for NaNs and let them poison the average (audit bug #2 --
    verified to affect ~65% of rows for the detailed-stats columns).
    """
    if not history_list:
        return {}

    avgs: dict = {}
    keys = history_list[0].keys()

    for k in keys:
        values = [match[k] for match in history_list if pd.notna(match[k])]
        avgs[f"avg_{k}"] = np.mean(values) if values else 0

    if avgs.get("avg_svpt", 0) > 0:
        avgs["pct_1stIn"] = avgs.get("avg_1stIn", 0) / avgs.get("avg_svpt")
        avgs["pct_1stWon"] = (
            avgs.get("avg_1stWon", 0) / avgs.get("avg_1stIn")
            if avgs.get("avg_1stIn", 0) > 0
            else 0
        )
        second_serve_pts = avgs.get("avg_svpt") - avgs.get("avg_1stIn")
        avgs["pct_2ndWon"] = (
            avgs.get("avg_2ndWon", 0) / second_serve_pts if second_serve_pts > 0 else 0
        )

    wins = sum(m.get("won", 0) for m in history_list)
    avgs["recent_win_pct"] = wins / len(history_list)
    return avgs


def general_history(history_dict: dict, player: dict) -> dict:
    """Average stats for ``player`` over their recorded history window."""
    return get_avg_stats(history_dict.get(player["id"], []))


def h2h_history(history_dict: dict, player1: dict, player2: dict):
    """Average stats and win rate for the head-to-head between two players."""
    player1_id = player1["id"]
    player2_id = player2["id"]

    h2h_key = tuple(sorted([player1_id, player2_id]))
    matchup_history = history_dict.get(h2h_key, {})

    player1_h2h_list = matchup_history.get(player1_id, [])
    player2_h2h_list = matchup_history.get(player2_id, [])

    player1_h2h_stats = get_avg_stats(player1_h2h_list)
    player2_h2h_stats = get_avg_stats(player2_h2h_list)

    matches_num = len(player1_h2h_list)
    if matches_num > 0:
        p1_wins = sum(m.get("won", 0) for m in player1_h2h_list)
        p1_win_pct = p1_wins / matches_num
    else:
        p1_win_pct = 0.5  # Neutral value if they never played

    return player1_h2h_stats, player2_h2h_stats, matches_num, p1_win_pct


def update_general_history(history_dict: dict, player: dict, new_stats: dict, max_len: int | None = None):
    """Append ``new_stats`` to a player's rolling history, trimmed to ``max_len``."""
    player_id = player["id"]
    history_dict.setdefault(player_id, []).append(new_stats)
    if max_len is not None and len(history_dict[player_id]) > max_len:
        history_dict[player_id].pop(0)


def update_h2h_history(history_dict: dict, player1: dict, player2: dict, player1_new_stats: dict, player2_new_stats: dict):
    """Append the just-played match's stats to the two players' H2H history."""
    player1_id = player1["id"]
    player2_id = player2["id"]

    h2h_match = tuple(sorted([player1_id, player2_id]))
    matchup = history_dict.setdefault(h2h_match, {player1_id: [], player2_id: []})
    matchup.setdefault(player1_id, []).append(player1_new_stats)
    matchup.setdefault(player2_id, []).append(player2_new_stats)


def get_days_rest(last_match_date: dict, player: dict, current_date) -> int:
    """Days since ``player``'s last recorded match, clamped to ``[0, 60]``."""
    player_id = player["id"]
    if player_id in last_match_date:
        delta = (current_date - last_match_date[player_id]).days
        if delta < 0:
            return 30
        if delta > 60:
            return 60
        return delta
    return 30  # Neutral value for a player's first recorded match
