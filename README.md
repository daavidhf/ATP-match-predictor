# ATP Tennis Match Predictor

A machine learning project that predicts the winner of professional ATP
tennis matches, using a Random Forest classifier over hand-engineered,
strictly pre-match features: Elo (general and per-surface), rolling
recent form, head-to-head history, and match/tournament context.

The project started as a single 53-cell notebook. It was audited by
reconstructing the pipeline outside the notebook to measure the real
effect of each step, which surfaced four data-correctness bugs and a
generous headline number. The bugs are fixed, the pipeline now lives
in tested `src/` modules, and the numbers below are the honest,
post-fix ones.

## Results (2022 season holdout)

Trained on matches before December 2021, tested on the 2022 season
(2,927 matches) -- a fixed, chronological, single holdout, `src/`
default config.

| | Accuracy | Log-loss | Brier score |
|---|---|---|---|
| Baseline ("higher ATP rank wins") | 64.81% | -- | -- |
| Random Forest (corrected) | **67.03%** | 0.597 | 0.206 |

That's a **+2.2 point** improvement over the trivial baseline. Before
the Elo-leak fix, the same holdout showed accuracy around 69%, a
number that included ~2 points of look-ahead bias (see [Bugs
fixed](#bugs-fixed-in-this-audit) below) -- the honest improvement is
smaller than it first looked, which is why this project also reports
log-loss and Brier score rather than accuracy alone.

Reproduce with:

```bash
python scripts/run_pipeline.py
```

which also prints feature importances and a walk-forward
season-by-season cross-validation (the "official" validation strategy
for any decisions about features or hyperparameters -- see
[Methodology](#methodology)).

## Repository layout

```
atp-match-predictor/
├── data/
│   └── raw/                 # ATP CSVs (gitignored -- see Setup)
├── scripts/
│   ├── download_data.py     # refresh data from JeffSackmann/tennis_atp
│   └── run_pipeline.py      # build dataset, train, evaluate end-to-end
├── src/
│   ├── data.py               # raw CSV loading
│   ├── elo.py                 # Elo ratings (pre-match, general + per-surface)
│   ├── features.py            # rolling stats, H2H, fatigue, seed
│   ├── dataset.py             # P1-vs-P2 dataset construction
│   ├── train.py                # model, time split, walk-forward CV
│   ├── evaluate.py             # accuracy, log-loss, Brier, baseline
│   ├── predict.py              # inference on hypothetical matches
│   └── config.py               # config.yaml loader
├── tests/                    # unit tests, incl. regression tests for
│                              # each bug fixed in this audit
├── notebooks/
│   └── exploration.ipynb     # original notebook -- EDA/history only,
│                              # no longer the source of truth
├── config.yaml                 # window size, k-factor, cutting date, hyperparameters
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The bundled dataset (`data/raw/*.csv`, from [Jeff Sackmann's ATP
data](https://github.com/JeffSackmann/tennis_atp)) is gitignored, not
committed. Get it either by asking whoever has the repo checked out
locally for the three CSVs, or by refreshing from source:

```bash
python scripts/download_data.py --start-year 2000 --end-year 2024 --players --rankings
```

Then run the pipeline:

```bash
python scripts/run_pipeline.py
```

Run the tests (no data required -- they use small synthetic fixtures):

```bash
pytest
```

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
  end-of-project number.
* **Probabilities, not just a hard label.** This is meant to be a
  predictor of *probabilities* (for comparing against betting odds,
  see `predict.get_real_probabilities`), so log-loss and Brier score
  are reported alongside accuracy.
* **No explicit imputation.** Random Forest in scikit-learn 1.4+
  handles `NaN` inputs natively, which is what makes an unimputed
  feature matrix "work" here -- it isn't a deliberate design decision,
  and a model that doesn't have native NaN support (logistic
  regression, most other libraries) would need one. Worth adding
  explicit imputation, or comparing against a gradient-boosting model
  that documents its own NaN handling (XGBoost/LightGBM), as follow-up
  work.

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

* Only one bundled season cutoff (through 2022); refresh with
  `scripts/download_data.py`.
* No model comparison yet against gradient boosting (XGBoost/LightGBM)
  or a logistic-regression baseline.
* No explicit missing-value imputation (see Methodology above).
* `data/raw/*.csv` and `Dataset/Data.xlsx` (an unused 70MB file, now
  removed) were committed to git history before this reorganization;
  stopping tracking going forward doesn't shrink existing history --
  that would need a separate, explicit `git filter-repo`/BFG pass.

---
*Created by David*
