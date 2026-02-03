# ATP Tennis Match Predictor

A Machine Learning project designed to forecast the outcome of professional ATP tennis matches. This model uses a **Random Forest Classifier** and rigorous feature engineering to outperform baseline ranking predictions.

## Key Features

* **Leak-Free Validation:** Implements a strict `season_splitter` (Walk-Forward Validation) to ensure the model never trains on future data.
* **Dynamic Feature Engineering:**
    * **Rolling Stats:** Calculates player form (Serve %, Break Points Saved, Aces) based on a sliding window of the last 10 matches.
    * **Contextual H2H:** Logic to analyze Head-to-Head matchups specifically by surface and recency, not just raw averages.
    * **Surface Awareness:** Adapts predictions based on Clay, Grass, or Hard Court performance.
* **Data Handling:** Robust handling of missing values (NaNs) and categorical encoding for rounds and tournament levels.

## Tech Stack

* **Python 3.12+**
* **Pandas & NumPy:** Data manipulation and vectorization.
* **Scikit-Learn:** Random Forest implementation and model evaluation.

## Methodology

The model avoids the common pitfall of using "match statistics" (like duration or score) as features. Instead, it reconstructs the state of knowledge **before** the match begins, using historical dictionaries to simulate real-time betting scenarios.

## Performance

* **Metric:** Accuracy
* **Validation Strategy:** Trained on past seasons (e.g., 2018-2021) and validated on subsequent full season.
* **Benchmark:** Designed to exceed the "Higher Rank Wins" baseline (>65%).

---
*Created by David*
