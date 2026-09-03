"""Regression test for audit bug #3: the ``pct_2nWon`` typo.

The notebook read the second-serve-points-won percentage back out of
the per-player stats dict with ``.get('pct_2nWon')`` (missing the
"d"), but ``get_avg_stats`` always produces the key ``pct_2ndWon``. So
``p1_pct_2ndWon``/``p2_pct_2ndWon``/``p1_h2h_pct_2ndWon``/
``p2_h2h_pct_2ndWon`` were always ``None``, no matter how much history
a player had. This test builds two matches -- the second with real
prior history -- and checks the column actually gets populated.
"""
import pandas as pd

from src.dataset import build_dataset


def _match_row(date, winner_id, loser_id, surface="Hard"):
    return {
        "tourney_date": date,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "surface": surface,
        "round": "R32",
        "tourney_level": "A",
        "best_of": 3,
        "draw_size": 32,
        "minutes": 90,
        "winner_seed": None,
        "winner_rank": 10,
        "winner_rank_points": 1000,
        "winner_age": 25.0,
        "winner_hand": "R",
        "winner_ht": 185.0,
        "winner_ioc": "ESP",
        "loser_seed": None,
        "loser_rank": 50,
        "loser_rank_points": 300,
        "loser_age": 24.0,
        "loser_hand": "R",
        "loser_ht": 180.0,
        "loser_ioc": "FRA",
        "w_ace": 5,
        "w_df": 2,
        "w_svpt": 60,
        "w_1stIn": 40,
        "w_1stWon": 25,
        "w_2ndWon": 10,
        "w_SvGms": 10,
        "w_bpSaved": 3,
        "w_bpFaced": 5,
        "l_ace": 3,
        "l_df": 4,
        "l_svpt": 65,
        "l_1stIn": 35,
        "l_1stWon": 20,
        "l_2ndWon": 12,
        "l_SvGms": 10,
        "l_bpSaved": 2,
        "l_bpFaced": 6,
    }


def _matches_with_history():
    rows = [
        _match_row("2020-01-01", winner_id=10, loser_id=20),
        # Player 10 plays again a month later: general_history now has
        # one prior match on record for them, so avg_svpt/avg_1stIn are
        # real numbers and pct_2ndWon should compute to something.
        _match_row("2020-02-01", winner_id=30, loser_id=10),
    ]
    df = pd.DataFrame(rows)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    return df


def test_pct_2ndwon_is_populated_when_history_exists():
    df_train, _ = build_dataset(_matches_with_history(), window_size=10, seed=1)

    second_row = df_train.iloc[1]
    populated = [
        v for v in (second_row["p1_pct_2ndWon"], second_row["p2_pct_2ndWon"]) if pd.notna(v)
    ]
    assert len(populated) == 1, "exactly one side (player 10) should have prior history"
    assert populated[0] > 0

    # The player with no history yet (first-ever match) must still be None.
    first_row = df_train.iloc[0]
    assert pd.isna(first_row["p1_pct_2ndWon"])
    assert pd.isna(first_row["p2_pct_2ndWon"])
