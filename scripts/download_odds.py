"""Download historical ATP betting odds from tennis-data.co.uk.

This is the data source the value-betting backtest (src/backtest.py)
actually needs and doesn't have a substitute for: it's the only free,
actively updated source found during this audit that carries
bookmaker odds (multiple books, including Pinnacle) alongside
match-level results. It does *not* carry the detailed serve/return
stats (aces, double faults, break points, ...) that src/features.py's
rolling-form features use -- for that, see scripts/download_data.py.

One file per season, ``http://www.tennis-data.co.uk/{year}/{year}.xlsx``.
This project does not yet merge these odds files with the Sackmann-derived
match dataset (src/data.load_matches): tennis-data.co.uk identifies
players by name, not by the Sackmann `player_id` this codebase uses
everywhere else, so a name-matching step is required first. That's
tracked as follow-up work -- see the README's Known limitations.

Note: this repo was authored in a sandboxed session whose network
policy blocks tennis-data.co.uk outright (confirmed via a 403 on the
CONNECT tunnel), even though the site itself is reported live and
updated -- this script could not be exercised end to end here. Run it
somewhere with normal internet access and sanity-check the first
download (open one file, confirm it has the expected columns:
Date, Winner, Loser, and odds columns like B365W/B365L, PSW/PSL for
Pinnacle, etc. -- the exact bookmaker columns have changed over the
years).

Usage:
    python scripts/download_odds.py --start-year 2000 --end-year 2024
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "http://www.tennis-data.co.uk"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "odds"


def download_season(year: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{year}/{year}.xlsx"
    dest = out_dir / f"{year}.xlsx"
    print(f"  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print("Downloading odds files from tennis-data.co.uk...")
    failures = []
    for year in range(args.start_year, args.end_year + 1):
        try:
            download_season(year, out_dir)
        except Exception as exc:  # noqa: BLE001 -- keep going across seasons
            print(f"  failed for {year}: {exc}", file=sys.stderr)
            failures.append(year)

    if failures:
        print(f"\nDone with {len(failures)} failures: {failures}", file=sys.stderr)
    else:
        print("\nDone.")
    print(
        "Check tennis-data.co.uk's own terms of use before redistributing "
        "this data -- it's a separate source from the Sackmann archive "
        "(which is CC BY-NC-SA) and has its own licensing."
    )


if __name__ == "__main__":
    sys.exit(main())
