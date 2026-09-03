# ATP Tennis Match Predictor

A machine learning project that estimates a calibrated win probability
for professional ATP tennis matches, using a Random Forest over
hand-engineered, strictly pre-match features: Elo (general and
per-surface), rolling recent form, head-to-head history, and
match/tournament context.

**The goal is not to maximize accuracy.** It's to produce a
well-calibrated win probability and compare it against the betting
market's own (de-vigged) probability, to see whether the two disagree
in a way that would have been profitable to bet on ("value betting").
Accuracy is kept as a cheap sanity check that the model learned real
signal; it is not the number this project optimizes for or reports as
its bottom line. See [Objective: calibration and value
betting](#objective-calibration-and-value-betting) below.

The project started as a single 53-cell notebook. It was audited by
reconstructing the pipeline outside the notebook to measure the real
effect of each step, which surfaced four data-correctness bugs and a
generous headline accuracy number. The bugs are fixed, the pipeline
now lives in tested `src/` modules, and the numbers below are the
honest, post-fix ones.

> **Not financial advice.** A backtest can show whether there was
> historical edge; it does not guarantee future profit. ATP tour-level
> tennis betting markets, especially at sharp books like Pinnacle, are
> highly efficient.

## Objective: calibration and value betting

The project's actual success criterion is: does the model's win
probability, compared against the market's de-vigged probability,
identify bets with positive expected value -- repeated across many
matches, not on one lucky pick?

That changes what matters at every stage:

* **Calibration over accuracy.** If the model says 60%, it should win
  about 60% of the time. Log-loss, Brier score, and a reliability
  (calibration) table are reported; accuracy is not the headline
  metric (see [Results](#results-2022-season-holdout)).
* **The real baseline is the closing line, not "higher rank wins."**
  The rank-baseline numbers below are a sanity check that the model
  learned something -- they are not the bar that decides whether this
  project works. That bar is whether the model's probability diverges
  from the market's own **closing** odds (not opening odds -- using
  opening odds would let the backtest "see" the line movement that
  happened after the model would have had to place its bet, which is
  its own flavor of data leakage).
* **A backtest simulates actual bets**, not just probability
  comparisons: a staking strategy (flat or fractional Kelly), P&L
  replayed in chronological order (never shuffled), and reported as
  ROI and max drawdown -- see `src/backtest.py`.
* **The edge threshold is chosen walk-forward, not against the
  reported holdout.** Same discipline as `season_splitter`: pick
  which edge size is worth betting on using validation seasons, and
  score that one choice, once, against an untouched final season. See
  `backtest.choose_threshold_and_evaluate_holdout`.
* De-vigging odds today uses simple proportional normalization
  (`src/odds.py`); Shin's method is a documented, more accurate
  correction for the favourite-longshot bias, tracked as follow-up
  work (see [Known limitations](#known-limitations--follow-up)).

**Status:** `src/odds.py` and `src/backtest.py` are built and tested
against synthetic data, but **this repo does not yet contain real
historical closing odds**, so the backtest hasn't been run against
real markets yet -- see [Data sources](#data-sources) for why, and
what's needed to close that gap.

## Results (2022 season holdout)

Trained on matches before December 2021, tested on the 2022 season
(2,927 matches) -- a fixed, chronological, single holdout, `src/`
default config.

| | Log-loss | Brier score | Accuracy (sanity check) |
|---|---|---|---|
| Baseline ("higher ATP rank wins") | -- | -- | 64.81% |
| Random Forest (corrected) | 0.597 | 0.206 | **67.03%** |

The model beats the rank baseline by +2.2 accuracy points -- real
signal, but that's not the number that matters here (see
[Objective](#objective-calibration-and-value-betting) above). Before
the Elo-leak fix, the same holdout showed accuracy around 69%, which
included ~2 points of look-ahead bias (see [Bugs
fixed](#bugs-fixed-in-this-audit) below).

Reproduce with:

```bash
python scripts/run_pipeline.py
```

which prints log-loss/Brier/accuracy, a calibration table, feature
importances, and a walk-forward season-by-season cross-validation
(the "official" validation strategy for feature/hyperparameter
decisions -- see [Methodology](#methodology)). That walk-forward CV,
replaying all 54 seasons from 1969 through 2022 (expanding training
window, testing on the season immediately after), gives a **mean
accuracy of 68.9%** (std 2.9 pts across seasons) with the last, 2022
split at 66.8% -- consistent with the single fixed 2022 holdout above.

**Calibration** (10 quantile buckets, predicted vs. observed P1-win
rate on the 2022 holdout):

| Predicted | Observed |
|---|---|
| 0.157 | 0.157 |
| 0.275 | 0.270 |
| 0.352 | 0.339 |
| 0.418 | 0.461 |
| 0.474 | 0.396 |
| 0.527 | 0.497 |
| 0.583 | 0.539 |
| 0.648 | 0.644 |
| 0.726 | 0.768 |
| 0.836 | 0.881 |

Reasonably well calibrated overall (most buckets track the diagonal
within a few points), with the widest gaps in the 0.42-0.53
mid-probability range (~6-8 points off) -- exactly the kind of
systematic wobble that either points to a real, exploitable market
disagreement or to the model needing more work there. This is a
single 2,927-match holdout, not a calibration curve validated across
multiple seasons -- treat it as a first read, not a final verdict.

## Repository layout

```
atp-match-predictor/
├── data/
│   └── raw/                 # ATP CSVs (gitignored -- see Setup)
├── scripts/
│   ├── download_data.py     # refresh match/player data (see Data sources)
│   ├── download_odds.py     # historical betting odds from tennis-data.co.uk
│   └── run_pipeline.py      # build dataset, train, evaluate end-to-end
├── src/
│   ├── data.py               # raw CSV loading
│   ├── elo.py                 # Elo ratings (pre-match, general + per-surface)
│   ├── features.py            # rolling stats, H2H, fatigue, seed
│   ├── dataset.py             # P1-vs-P2 dataset construction
│   ├── train.py                # model, time split, walk-forward CV
│   ├── evaluate.py             # log-loss, Brier, calibration (accuracy is secondary)
│   ├── odds.py                  # de-vigging bookmaker odds
│   ├── backtest.py              # staking simulation, ROI, drawdown, threshold selection
│   ├── predict.py              # inference on hypothetical matches
│   └── config.py               # config.yaml loader
├── tests/                    # unit tests, incl. regression tests for
│                              # each bug fixed in this audit
├── notebooks/
│   └── exploration.ipynb     # original notebook -- EDA/history only,
│                              # no longer the source of truth
├── config.yaml                 # window size, k-factor, cutting date, model + backtest params
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The bundled dataset (`data/raw/*.csv`) is gitignored, not committed.
See [Data sources](#data-sources) below for where to get it and why
that's more complicated than it should be. Once you have it:

```bash
python scripts/run_pipeline.py
```

Run the tests (no data required -- they use small synthetic fixtures):

```bash
pytest
```

## Data sources

The original source, `JeffSackmann/tennis_atp` on GitHub, **is gone**
-- confirmed 404 during this audit (the account exists, the repo
doesn't). There is no official ATP bulk-data source either:
`atptour.com` doesn't publish one, and the "official" feeds
(Infosys/Deltatre) are licensed commercially to broadcasters and
sportsbooks, not available for a personal project. None of the
alternatives found give everything the original did (rich stats,
live updates, odds, free) at once -- something has to be traded off:

1. **A frozen mirror of the Sackmann archive**
   ([`Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)
   on Hugging Face). Same detailed columns (aces, double faults, break
   points) this project's features already depend on, but not updated
   past ~2024. `scripts/download_data.py` targets this.
2. **[tennis-data.co.uk](http://www.tennis-data.co.uk)**. Live and
   actively updated, with per-season odds files from several
   bookmakers including Pinnacle -- and since the project's actual
   goal is comparing against market odds, **this source stops being
   optional**: it's the only free source found that has odds at all.
   It does **not** have aces/double-faults/serve-points/break-points,
   so the `get_avg_stats` features built on those columns can't come
   from here alone. `scripts/download_odds.py` targets this.
3. **Commercial APIs** (Sportradar, API-Tennis, etc.) -- complete and
   current, but paid; check licensing for personal/non-commercial use
   before relying on one.

**Chosen approach: hybrid.** Train on the rich Sackmann-derived
history (mirror, through ~2024) for the detailed serve/return/rolling
features, and pull `tennis-data.co.uk` specifically for closing odds
to power the backtest. **This merge is not implemented yet** --
`tennis-data.co.uk` identifies players by name, this codebase's
Sackmann-derived pipeline identifies them by `player_id`, and there's
no name-matching step between the two yet. That's the concrete
next piece of work to actually run `src/backtest.py` against real
markets (see [Known limitations](#known-limitations--follow-up)).

Any data derived from the Sackmann archive (the bundled CSVs, and the
Hugging Face mirror) carries its **CC BY-NC-SA license: attribution
required, non-commercial use only.** Keep that in mind before using
this project's data or model commercially. `tennis-data.co.uk` is a
separate source with its own terms -- check them before redistributing
that data.

**Sandboxing note:** this repo was reorganized in a sandboxed Claude
Code session whose network policy blocks both `huggingface.co` and
`tennis-data.co.uk` outright (confirmed via 403s on the CONNECT
tunnel), so neither `scripts/download_data.py`'s Hugging Face path nor
`scripts/download_odds.py` could be exercised end-to-end from inside
that session -- only the (now-confirmed-dead) original Sackmann repo
URL could actually be reached and tested from there. Run both scripts
somewhere with normal internet access and sanity-check the first
download of each.

## Methodology

* **No leakage by construction.** Every feature is built from state
  known strictly *before* the match: Elo ratings are recorded
  pre-update, rolling stats come from a player's last `WINDOW_SIZE`
  matches *prior* to this one, and the winner/loser columns are
  reshuffled into a neutral Player A vs Player B framing (with a
  seeded random assignment) so the model can't learn "the winner
  column always wins."
* **Chronological split, not random.** `train_test_split` would let
  the model train on 2022 matches and get tested on 2021 ones. The
  primary holdout instead trains on everything before December 2021
  and tests on the full 2022 season.
* **Walk-forward CV is the real validation.** A single fixed holdout,
  reused repeatedly to decide which features to drop, quietly
  overfits to that one holdout. `season_splitter` (in `src/train.py`)
  instead does an expanding-window, season-by-season split -- train on
  every season so far, test on the next one, repeat -- and should
  drive any feature or hyperparameter decisions. The single December
  2021 / 2022 holdout above is kept untouched as the final,
  end-of-project number. The same discipline applies to picking a
  backtest edge threshold -- see `backtest.choose_threshold_and_evaluate_holdout`.
* **Closing odds, not opening odds, for any backtest.** Comparing
  against opening odds inflates the apparent edge, because the line
  moves on information the model didn't have at prediction time --
  a domain-specific form of leakage.
* **No explicit imputation.** Random Forest in scikit-learn 1.4+
  handles `NaN` inputs natively, which is what makes an unimputed
  feature matrix "work" here -- it isn't a deliberate design decision,
  and a model without native NaN support (logistic regression, most
  other libraries) would need one. Worth adding explicit imputation,
  or comparing against a gradient-boosting model that documents its
  own NaN handling (XGBoost/LightGBM), evaluated with the same
  log-loss/Brier metrics rather than accuracy alone.

## Bugs fixed in this audit

The pipeline was reconstructed outside the notebook to measure the
before/after effect of each fix on the same 2022 holdout. In the order
they were fixed:

1. **Elo data leak (`src/elo.py`).** `w_elo_gen`/`l_elo_gen` were
   computed *after* the main Elo loop by mapping the *final*
   `elo_general` dict onto every row -- applying each player's
   end-of-history rating to all of their past matches. Surface Elo
   didn't have this bug (it was already captured inside the loop).
   Fixing it dropped test accuracy from ~69% to 67.0% -- a ~2 point
   improvement that wasn't real -- and flipped the feature-importance
   ranking: `diff_elo_gen` now outranks `diff_elo_surf`, which makes
   more sense than the reverse.
2. **NaN identity bug (`src/features.py`, `get_avg_stats`).**
   `match[k] is not np.nan` compares object identity, not whether a
   value is missing. A NaN read out of a pandas cell is never the
   same object as the `np.nan` singleton, so the filter was true even
   for NaNs and let them into the rolling averages. This corrupted
   `p1_ace`, `p1_df`, `p1_1stWon`, `p1_bpSaved` (and their `p2_`/H2H
   counterparts) in roughly **65% of rows** (~123,000 of 188,161).
   Fixed with `pd.notna(...)`.
3. **`pct_2ndWon` typo (`src/dataset.py`).** The dataset builder read
   the second-serve-win-rate back out with `.get('pct_2nWon')`
   (missing the "d") in all four places it's used
   (`p1_pct_2ndWon`, `p2_pct_2ndWon`, `p1_h2h_pct_2ndWon`,
   `p2_h2h_pct_2ndWon`), but `get_avg_stats` actually produces the key
   `pct_2ndWon`. All four columns were silently `None` for every row.
4. **Surface Elo lookup in inference (`src/predict.py`,
   `get_player_info`).** `elo_surface.get(player_id, 1500)` looked a
   player id up directly in a dict keyed by *surface*
   (`{'Hard': {...}, 'Clay': {...}, ...}`), which can never match --
   so the prediction demo always used the 1500 default, regardless of
   the player or the match surface, despite surface Elo being the
   project's headline feature. Fixed to look the player up inside the
   surface's own sub-dict, given the match surface as a parameter.
   (While in there: `get_player_info` used to re-read the full
   players and rankings CSVs from disk on every call -- now callers
   load them once and pass them in.)

Regression tests for all four live in `tests/`.

## Known limitations / follow-up

* **No real odds data merged in yet.** `src/odds.py` and
  `src/backtest.py` are implemented and unit-tested against synthetic
  data, but running an actual value-betting backtest needs
  `tennis-data.co.uk` odds joined to the Sackmann-derived match
  dataset by date + player name (no shared player id between the two
  sources) -- that name-matching step doesn't exist yet. This is the
  single most important piece of unfinished work given the project's
  actual goal.
* **De-vigging is simple normalization, not Shin's method.** Simple
  normalization distorts the favourite-longshot bias; Shin's method is
  a documented improvement, not yet implemented (see `src/odds.py`).
* Only one bundled season cutoff (through ~2022-2024 depending on
  source); refresh with `scripts/download_data.py`.
* No model comparison yet against gradient boosting (XGBoost/LightGBM)
  or a logistic-regression baseline, evaluated on log-loss/Brier
  rather than accuracy.
* No explicit missing-value imputation (see Methodology above).
* `data/raw/*.csv` and `Dataset/Data.xlsx` (an unused 70MB file, now
  removed) were committed to git history before this reorganization;
  stopping tracking going forward doesn't shrink existing history --
  that would need a separate, explicit `git filter-repo`/BFG pass.

---
*Created by David*
