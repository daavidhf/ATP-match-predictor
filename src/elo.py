"""Elo rating calculation for ATP matches.

Both the general Elo and the per-surface Elo are recorded strictly
*before* a match is played, and the ratings are only updated *after*
the pre-match values have been captured and returned. This is what
guarantees no future information leaks into a match's features.

(Audit bug #1: the notebook computed ``w_elo_gen``/``l_elo_gen`` after
the main loop, by applying the *final* ``elo_general`` dict to every
row via ``.apply(...)``. That put a player's end-of-history rating on
every one of their past matches -- future information leaking into
past rows. The surface Elo did not have this problem because it was
already captured inside the loop. ``compute_elo`` below records both
ratings inside the loop, the same way, so neither can leak.)
"""
from __future__ import annotations

import pandas as pd

DEFAULT_ELO = 1500.0
DEFAULT_SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that a player rated ``rating_a`` beats one rated ``rating_b``."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


class EloRatings:
    """Tracks general and per-surface Elo ratings as matches are replayed."""

    def __init__(self, k_factor: float = 32, surfaces=DEFAULT_SURFACES):
        self.k_factor = k_factor
        self.general: dict = {}
        self.surface: dict = {s: {} for s in surfaces}

    def get_general(self, player_id) -> float:
        return self.general.get(player_id, DEFAULT_ELO)

    def get_surface(self, player_id, surface) -> float:
        return self.surface.get(surface, {}).get(player_id, DEFAULT_ELO)

    def get(self, player_id, surface) -> float:
        """Surface Elo if the player has one tracked, else their general Elo."""
        surf_ratings = self.surface.get(surface)
        if surf_ratings is not None and player_id in surf_ratings:
            return surf_ratings[player_id]
        return self.get_general(player_id)

    def update(self, winner_id, loser_id, surface):
        """Record the result of a finished match and update ratings.

        Returns the *pre-match* ``(w_elo_surf, l_elo_surf, w_elo_gen,
        l_elo_gen)`` tuple, i.e. what a model may legitimately use as
        a feature for this match.
        """
        w_elo_surf = self.get_surface(winner_id, surface)
        l_elo_surf = self.get_surface(loser_id, surface)
        w_elo_gen = self.get_general(winner_id)
        l_elo_gen = self.get_general(loser_id)

        if surface in self.surface:
            exp_w_surf = expected_score(w_elo_surf, l_elo_surf)
            exp_l_surf = expected_score(l_elo_surf, w_elo_surf)
            self.surface[surface][winner_id] = w_elo_surf + self.k_factor * (1 - exp_w_surf)
            self.surface[surface][loser_id] = l_elo_surf + self.k_factor * (0 - exp_l_surf)

        exp_w_gen = expected_score(w_elo_gen, l_elo_gen)
        exp_l_gen = expected_score(l_elo_gen, w_elo_gen)
        self.general[winner_id] = w_elo_gen + self.k_factor * (1 - exp_w_gen)
        self.general[loser_id] = l_elo_gen + self.k_factor * (0 - exp_l_gen)

        return w_elo_surf, l_elo_surf, w_elo_gen, l_elo_gen


def compute_elo(matches: pd.DataFrame, k_factor: float = 32, surfaces=DEFAULT_SURFACES):
    """Replay a chronologically-sorted match dataframe and attach pre-match Elo.

    Requires ``winner_id``, ``loser_id`` and ``surface`` columns. Returns
    ``(matches_with_elo, ratings)`` where ``matches_with_elo`` has four
    new columns (``w_elo_surf``, ``l_elo_surf``, ``w_elo_gen``,
    ``l_elo_gen``, all pre-match) and ``ratings`` is the ``EloRatings``
    instance holding the *final* ratings after every match -- useful
    for scoring a brand-new, hypothetical match at inference time.
    """
    ratings = EloRatings(k_factor=k_factor, surfaces=surfaces)

    w_elo_surf, l_elo_surf, w_elo_gen, l_elo_gen = [], [], [], []
    for row in matches.itertuples(index=False):
        ws, ls, wg, lg = ratings.update(row.winner_id, row.loser_id, row.surface)
        w_elo_surf.append(ws)
        l_elo_surf.append(ls)
        w_elo_gen.append(wg)
        l_elo_gen.append(lg)

    matches = matches.copy()
    matches["w_elo_surf"] = w_elo_surf
    matches["l_elo_surf"] = l_elo_surf
    matches["w_elo_gen"] = w_elo_gen
    matches["l_elo_gen"] = l_elo_gen
    return matches, ratings
