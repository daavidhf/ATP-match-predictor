from src.odds import devig_normalize, get_real_probabilities, implied_probability


def test_implied_probability():
    assert implied_probability(2.0) == 0.5
    assert round(implied_probability(4.0), 4) == 0.25


def test_devig_normalize_removes_the_margin():
    # Symmetric market with a margin: raw implied probabilities sum to
    # > 1 (1/1.8 + 1/1.8 = 1.111...), de-vigged they must sum to
    # exactly 1, and stay 50/50 since both sides are priced the same.
    p1, p2 = devig_normalize(1.8, 1.8)
    assert round(p1 + p2, 9) == 1.0
    assert round(p1, 6) == 0.5
    assert round(p2, 6) == 0.5


def test_devig_normalize_asymmetric_market():
    # A heavy favourite (1.2) against a big underdog (5.0): raw implied
    # probabilities are 0.8333 and 0.2, summing to 1.0333 (a ~3.3%
    # margin). De-vigged, they should keep the same *ratio* but sum to 1.
    p1, p2 = devig_normalize(1.2, 5.0)
    assert round(p1 + p2, 9) == 1.0
    assert p1 > p2
    raw_p1, raw_p2 = implied_probability(1.2), implied_probability(5.0)
    assert round(p1 / p2, 6) == round(raw_p1 / raw_p2, 6)


def test_get_real_probabilities_matches_devig_normalize_as_percentages():
    p1, p2 = devig_normalize(1.8, 1.8)
    pct1, pct2 = get_real_probabilities(1.8, 1.8)
    assert round(pct1, 6) == round(p1 * 100, 6)
    assert round(pct2, 6) == round(p2 * 100, 6)
