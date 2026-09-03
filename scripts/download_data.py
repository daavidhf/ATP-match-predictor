"""Download/refresh ATP match data from Jeff Sackmann's public dataset.

The bundled ``data/raw/atp_matches_till_2022.csv`` stops at the 2022
season. This pulls the yearly match files (and, optionally, players
and rankings) straight from
https://github.com/JeffSackmann/tennis_atp and concatenates them, so
the pipeline can be re-run against fresher data.

Usage:
    python scripts/download_data.py --start-year 2015 --end-year 2024
    python scripts/download_data.py --players --rankings
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw"


def _download(url: str, dest: Path):
    print(f"  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_matches(start_year: int, end_year: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        _download(f"{RAW_BASE}/atp_matches_{year}.csv", out_dir / f"atp_matches_{year}.csv")


def download_players(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _download(f"{RAW_BASE}/atp_players.csv", out_dir / "atp_players.csv")


def download_rankings(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in ("00s", "10s", "20s", "current"):
        _download(
            f"{RAW_BASE}/atp_rankings_{chunk}.csv",
            out_dir / f"atp_rankings_{chunk}.csv",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--players", action="store_true", help="Also download the players table")
    parser.add_argument("--rankings", action="store_true", help="Also download the rankings tables")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print("Downloading match files...")
    download_matches(args.start_year, args.end_year, out_dir)

    if args.players:
        print("Downloading players table...")
        download_players(out_dir)

    if args.rankings:
        print("Downloading rankings tables...")
        download_rankings(out_dir)

    print(
        "\nDone. Concatenate the yearly match files yourself (or extend "
        "src/data.load_matches) before feeding them to scripts/run_pipeline.py -- "
        "column layout is identical to the bundled atp_matches_till_2022.csv."
    )


if __name__ == "__main__":
    sys.exit(main())
