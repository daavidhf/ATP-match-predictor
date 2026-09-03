"""Regression tests for audit bug #1: the general-Elo data leak.

The notebook computed ``w_elo_gen``/``l_elo_gen`` *after* the main Elo
loop finished, by mapping the final ``elo_general`` dict onto every
row -- so a player's very first recorded match ended up carrying their
*final* career Elo. ``compute_elo`` must instead record the rating a
player had immediately before each of their matches.
"""
import pandas as pd

from src.elo import DEFAULT_ELO, EloRatings, compute_elo


def _matches(rows):
    df = pd.DataFrame(rows)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    return df.sort_values("tourney_date").reset_index(drop=True)


def test_first_match_uses_default_elo_not_final_elo():
    # Player 1 beats player 2 three times in a row on Hard.
    matches = _matches(
        [
            {"tourney_date": "2020-01-01", "winner_id": 1, "loser_id": 2, "surface": "Hard"},
            {"tourney_date": "2020-02-01", "winner_id": 1, "loser_id": 2, "surface": "Hard"},
            {"tourney_date": "2020-03-01", "winner_id": 1, "loser_id": 2, "surface": "Hard"},
        ]
    )
    out, ratings = compute_elo(matches)

    # Player 1's Elo going into their very first ever match must be the
    # untouched default -- not whatever their rating ended up being
    # after three wins.
    assert out.loc[0, "w_elo_gen"] == DEFAULT_ELO
    assert out.loc[0, "l_elo_gen"] == DEFAULT_ELO

    # After three wins, player 1's *final* general Elo has moved up.
    assert ratings.get_general(1) > DEFAULT_ELO
    # And that final rating must differ from what was recorded on the
    # first row -- if it didn't, the leak would still be there.
    assert out.loc[0, "w_elo_gen"] != ratings.get_general(1)


def test_general_elo_recorded_pre_match_each_row():
    matches = _matches(
        [
            {"tourney_date": "2020-01-01", "winner_id": 1, "loser_id": 2, "surface": "Hard"},
            {"tourney_date": "2020-02-01", "winner_id": 2, "loser_id": 1, "surface": "Clay"},
        ]
    )
    out, ratings = compute_elo(matches)

    # Row 2's pre-match Elo for player 1 (now losing) reflects their
    # win in row 1 -- but it must not be their *final* rating, which
    # also folds in the row-2 loss.
    elo_before_row2 = out.loc[1, "l_elo_gen"]
    assert elo_before_row2 != DEFAULT_ELO
    assert elo_before_row2 != ratings.get_general(1)


def test_surface_elo_falls_back_to_general_when_untracked():
    ratings = EloRatings(k_factor=32)
    assert ratings.get(99, "Grass") == DEFAULT_ELO

    ratings.update(winner_id=99, loser_id=100, surface="Hard")
    # Player 99 has never played on Grass, so the lookup falls back to
    # their general Elo (which moves on every match, regardless of
    # surface) rather than the raw 1500 default.
    assert ratings.get(99, "Grass") == ratings.get_general(99)
    assert ratings.get(99, "Grass") != DEFAULT_ELO
    # Their Hard-specific rating is tracked and reachable directly.
    assert ratings.get(99, "Hard") == ratings.surface["Hard"][99]
