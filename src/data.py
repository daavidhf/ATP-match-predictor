"""Raw ATP match data loading."""
from __future__ import annotations

import pandas as pd


def load_matches(path: str) -> pd.DataFrame:
    """Load the Sackmann ATP matches CSV, parsed and sorted chronologically."""
    matches = pd.read_csv(path)
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    return matches.sort_values("tourney_date").reset_index(drop=True)
