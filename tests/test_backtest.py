import pandas as pd

from src.backtest import (
    add_market_probabilities,
    choose_threshold_and_evaluate_holdout,
    compute_edges,
    kelly_fraction,
    select_bets,
    simulate_backtest,
)


def test_add_market_probabilities_sums_to_one():
    df = pd.DataFrame({"odds_p1": [1.8, 1.2], "odds_p2": [1.8, 5.0]})
    out = add_market_probabilities(df)
    assert (round(out["market_p1"] + out["market_p2"], 9) == 1.0).all()
    assert round(out.loc[0, "market_p1"], 6) == 0.5


def test_compute_edges_are_opposite():
    df = pd.DataFrame({"p1_win_prob": [0.7, 0.3], "odds_p1": [2.0, 2.0], "odds_p2": [2.0, 2.0]})
    out = compute_edges(df)
    assert round(out.loc[0, "edge_p1"], 6) == 0.2
    assert round(out.loc[0, "edge_p2"], 6) == -0.2
    assert (round(out["edge_p1"] + out["edge_p2"], 9) == 0.0).all()


def test_kelly_fraction_positive_edge():
    # b = odds - 1 = 1, q = 1 - 0.6 = 0.4 -> f* = (1*0.6 - 0.4) / 1 = 0.2
    assert round(kelly_fraction(0.6, 2.0, fraction=1.0), 6) == 0.2
    assert round(kelly_fraction(0.6, 2.0, fraction=0.5), 6) == 0.1


def test_kelly_fraction_negative_edge_clips_to_zero():
    # A model probability below the break-even (implied) probability
    # should never propose a positive stake.
    assert kelly_fraction(0.4, 2.0, fraction=1.0) == 0.0


def test_select_bets_picks_the_side_that_clears_the_threshold():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "p1_win_prob": [0.7, 0.3, 0.52],
            "odds_p1": [2.0, 2.0, 2.0],
            "odds_p2": [2.0, 2.0, 2.0],
            "target": [1, 1, 1],
        }
    )
    out = select_bets(df, edge_threshold=0.05)
    assert out.loc[0, "bet_side"] == "p1"
    assert out.loc[1, "bet_side"] == "p2"
    # No side clears the threshold -- pandas may store this "no bet"
    # marker as None or as NaN depending on version, so check for
    # either rather than assuming a specific missing-value sentinel.
    assert pd.isna(out.loc[2, "bet_side"])
    # Row 0: bet p1, target says p1 won -> bet won.
    assert bool(out.loc[0, "bet_won"]) is True
    # Row 1: bet p2, but target says p1 won -> bet lost.
    assert bool(out.loc[1, "bet_won"]) is False


def test_simulate_backtest_roi_and_drawdown():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "p1_win_prob": [0.7, 0.7],
            "odds_p1": [2.0, 2.0],
            "odds_p2": [2.0, 2.0],
            "target": [1, 0],  # first bet wins, second bet loses
        }
    )
    result = simulate_backtest(
        df, edge_threshold=0.05, stake_strategy="flat", flat_stake=1.0, initial_bankroll=100.0
    )
    assert result["num_bets"] == 2
    assert result["total_staked"] == 2.0
    assert result["total_profit"] == 0.0
    assert result["roi"] == 0.0
    assert result["final_bankroll"] == 100.0
    # Bankroll path: 100 -> 101 (win) -> 100 (loss). Peak is 101, so the
    # drawdown from that peak back to 100 is (101-100)/101.
    assert round(result["max_drawdown"], 6) == round(1 / 101, 6)


def test_choose_threshold_and_evaluate_holdout_does_not_peek_at_holdout():
    validation_rows = [
        {"date": pd.Timestamp("2020-01-01"), "season_id": 1, "p1_win_prob": 0.7,
         "odds_p1": 2.0, "odds_p2": 2.0, "target": 1},
        {"date": pd.Timestamp("2021-01-01"), "season_id": 2, "p1_win_prob": 0.7,
         "odds_p1": 2.0, "odds_p2": 2.0, "target": 1},
    ]
    holdout_row = {
        "date": pd.Timestamp("2022-01-01"), "season_id": 3, "p1_win_prob": 0.7,
        "odds_p1": 2.0, "odds_p2": 2.0, "target": 0,  # the value bet loses
    }
    df = pd.DataFrame(validation_rows + [holdout_row])

    result = choose_threshold_and_evaluate_holdout(
        df,
        thresholds=[0.05, 0.3],  # 0.3 clears no bets (edge is only 0.2)
        final_season=3,
        stake_strategy="flat",
        flat_stake=1.0,
        initial_bankroll=100.0,
    )

    # Only threshold 0.05 placed any bets in validation, so it must win.
    assert result["best_threshold"] == 0.05
    # Both validation bets won -> positive validation ROI.
    assert result["validation_roi"] > 0

    # The holdout result must reflect ONLY the holdout season (one lost
    # bet), not the winning validation bets -- if it did, ROI would be
    # positive instead of a total loss.
    holdout = result["holdout_result"]
    assert holdout["num_bets"] == 1
    assert holdout["roi"] == -1.0
