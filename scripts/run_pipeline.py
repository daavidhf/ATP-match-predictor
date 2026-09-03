"""End-to-end pipeline: build the dataset, train, and report honest metrics.

Usage:
    python scripts/run_pipeline.py [--config config.yaml] [--skip-walk-forward]

Prints, on the fixed 2022 holdout used throughout the project audit:
  * the trivial "higher rank wins" baseline,
  * the corrected Random Forest (accuracy, log-loss, Brier score),
  * per-tournament-level and per-round accuracy breakdowns,
  * feature importances,
and, unless ``--skip-walk-forward`` is passed, the walk-forward
season-by-season cross-validation accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import cross_val_score

from src.config import load_config
from src.data import load_matches
from src.dataset import build_dataset
from src.evaluate import baseline_higher_rank_wins, calibration_table, evaluate_predictions
from src.train import make_model, season_splitter, season_id, time_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--skip-walk-forward", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parent.parent

    t0 = time.time()
    print("Loading matches...")
    matches = load_matches(repo_root / config["data"]["matches_csv"])
    print(f"  {len(matches):,} matches loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Building dataset (Elo + rolling history features)...")
    df_train, _context = build_dataset(
        matches,
        window_size=config["features"]["window_size"],
        seed=config["dataset_seed"],
    )
    print(f"  {len(df_train):,} rows built in {time.time() - t0:.1f}s")

    X_train, y_train, X_test, y_test = time_split(df_train, config["split"]["cutting_date"])
    print(f"Training matches (history): {len(X_train):,}")
    print(f"Testing matches (2022 season): {len(X_test):,}")

    baseline_pred = baseline_higher_rank_wins(
        df_train[df_train["date"] >= pd.to_datetime(config["split"]["cutting_date"])]
    )
    baseline_acc = (baseline_pred == y_test.to_numpy()).mean()
    print(f"\nBaseline ('higher ATP rank wins') accuracy: {baseline_acc:.4f}")

    t0 = time.time()
    print("\nTraining Random Forest...")
    model = make_model(**config["model"])
    model.fit(X_train, y_train)
    print(f"  trained in {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    print("\n--- Test metrics (2022 holdout) -- log-loss/Brier are what matter, accuracy is a sanity check ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  improvement over baseline accuracy: {(metrics['accuracy'] - baseline_acc) * 100:+.2f} pts")
    print(
        "  (this baseline and accuracy improvement are a sanity check only -- "
        "the real bar is backtest ROI against de-vigged market odds, see src/backtest.py)"
    )

    print("\n--- Calibration (10 quantile buckets: predicted vs. observed P1-win rate) ---")
    print(calibration_table(y_test, y_proba).to_string(index=False))

    importances = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    print("\n--- Top 10 feature importances ---")
    print(importances.head(10).to_string())

    if not args.skip_walk_forward:
        t0 = time.time()
        print("\nRunning walk-forward season CV (this replays every season)...")
        df = df_train.copy()
        df["season_id"] = season_id(df["date"])
        X = df.drop(columns=["target", "date", "season_id"])
        y = df["target"]
        cv_model = make_model(n_estimators=200)
        scores = cross_val_score(cv_model, X, y, cv=season_splitter(df, "season_id"), scoring="accuracy")
        print(f"  {len(scores)} splits evaluated in {time.time() - t0:.1f}s")
        print(f"  mean walk-forward accuracy: {scores.mean():.4f} (std {scores.std():.4f})")
        print(f"  last (2022) split accuracy: {scores[-1]:.4f}")


if __name__ == "__main__":
    main()
