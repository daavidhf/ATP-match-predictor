"""De-vigging bookmaker odds into comparable probabilities.

Bookmaker odds embed a margin (the "vig"/overround): the implied
probabilities of both outcomes sum to *more* than 1. To judge whether
a model's probability actually disagrees with the market -- the whole
point of a value-betting backtest -- that margin has to be stripped
out first, so both sides are on the same 0-1 scale.
"""
from __future__ import annotations


def implied_probability(odds: float) -> float:
    """The naive, vig-inflated probability implied by a decimal odds price."""
    return 1 / odds


def devig_normalize(odd_p1: float, odd_p2: float) -> tuple[float, float]:
    """Remove the bookmaker's margin by proportional normalization.

    The simplest de-vig method: divide each side's implied probability
    by the sum of both sides' implied probabilities. This is what the
    original notebook's ``get_real_probabilities`` did.

    Known limitation: simple normalization distorts the
    favourite-longshot bias (it tends to over-correct big favourites
    and under-correct big underdogs). Shin's method is a documented,
    fairly simple improvement that models a small fraction of
    "insider" money to correct for this -- worth adding once the basic
    backtest pipeline is working end-to-end, but not implemented here
    yet (see README's Known limitations section).
    """
    implied_p1 = implied_probability(odd_p1)
    implied_p2 = implied_probability(odd_p2)
    total = implied_p1 + implied_p2
    return implied_p1 / total, implied_p2 / total


def get_real_probabilities(odd_p1: float, odd_p2: float) -> tuple[float, float]:
    """Same as ``devig_normalize``, as 0-100 percentages instead of 0-1.

    Kept for parity with the original notebook's one-off manual demo.
    """
    p1, p2 = devig_normalize(odd_p1, odd_p2)
    return p1 * 100, p2 * 100
