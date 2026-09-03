"""Regression test for audit bug #4: surface-Elo lookup in inference.

``elo_surface`` (now ``EloRatings.surface``) is keyed by surface
*first*, e.g. ``{'Hard': {player_id: elo}, 'Clay': {...}, ...}``. The
notebook's ``get_player_info`` looked a player up directly in that
outer dict with ``elo_surface.get(player_id, 1500)``, which can never
match a surface key against a numeric player id -- so it always fell
through to the 1500 default, regardless of who the player was or what
surface the hypothetical match was on. ``get_player_info`` must look
the player up *within* the given surface's sub-dict instead.
"""
import pandas as pd
import pytest

from src.elo import DEFAULT_ELO, EloRatings
from src.predict import get_player_info


@pytest.fixture
def players_df():
    return pd.DataFrame(
        [{"player_id": 1, "dob": pd.Timestamp("1995-04-27"), "hand": "R", "height": 193.0, "ioc": "AUS"}]
    ).set_index("player_id", drop=False)


@pytest.fixture
def rankings_df():
    return pd.DataFrame(
        [{"player": 1, "ranking_date": pd.Timestamp("2022-01-01"), "rank": 5, "points": 3000}]
    )


def test_get_player_info_uses_the_players_own_surface_elo(players_df, rankings_df):
    ratings = EloRatings(k_factor=32)
    # Give player 1 a real, moved-off-default Elo on Hard specifically.
    ratings.update(winner_id=1, loser_id=2, surface="Hard")
    hard_elo = ratings.get_surface(1, "Hard")
    assert hard_elo != DEFAULT_ELO

    info = get_player_info(1, players_df, rankings_df, ratings, surface="Hard", date="20220102")
    assert info["elo_surf"] == hard_elo
    assert info["elo_surf"] != DEFAULT_ELO


def test_get_player_info_falls_back_to_default_on_untracked_surface(players_df, rankings_df):
    ratings = EloRatings(k_factor=32)
    ratings.update(winner_id=1, loser_id=2, surface="Hard")

    # Player 1 has never played on Grass, so the surface Elo must be
    # the plain default -- not, say, their Hard rating leaking across.
    info = get_player_info(1, players_df, rankings_df, ratings, surface="Grass", date="20220102")
    assert info["elo_surf"] == DEFAULT_ELO
