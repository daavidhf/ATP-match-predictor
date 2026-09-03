"""Download/refresh ATP match history (player stats, no odds).

The bundled ``data/raw/atp_matches_till_2022.csv`` stops at the 2022
season and its original source is gone -- ``JeffSackmann/tennis_atp``
on GitHub now 404s (confirmed as of this audit; the account still
exists, the repo doesn't). There is no official ATP source either:
``atptour.com`` doesn't publish bulk data, and the "official" feeds
are licensed commercially (Infosys/Deltatre) to broadcasters and
sportsbooks, not available for a personal project.

The best remaining option for the detailed match stats this project's
features depend on (aces, double faults, serve/return points, break
points) is a frozen mirror of the same Sackmann archive:
https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive
-- same columns, but not updated past ~2024.

For closing betting odds specifically (needed for the value-betting
backtest, see src/backtest.py), see scripts/download_odds.py instead
-- tennis-data.co.uk is a *different*, actively updated source, but
does not carry the detailed serve/return stats this file's data does.
See the README's "Data sources" section for the full picture and the
CC BY-NC-SA attribution this data carries.

Note: this repo was authored in a sandboxed session whose network
policy blocks huggingface.co outright, so the mirror call below could
not be exercised end to end here -- sanity-check your first run.

Usage:
    pip install huggingface_hub   # only needed for this script
    python scripts/download_data.py --out-dir data/raw
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw"

DEAD_SACKMANN_RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
HF_MIRROR_REPO_ID = "Aneeshers/tennis-sackmann-archive"


def download_from_huggingface_mirror(out_dir: Path):
    """Fetch the frozen Sackmann-archive mirror from Hugging Face.

    Requires the ``huggingface_hub`` package. Downloads the dataset
    repo's full snapshot into ``out_dir`` -- inspect what lands there
    for the actual match/players/rankings file names, since this
    session couldn't reach huggingface.co to confirm the layout.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for this: pip install huggingface_hub"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading dataset repo '{HF_MIRROR_REPO_ID}' -> {out_dir}")
    snapshot_download(repo_id=HF_MIRROR_REPO_ID, repo_type="dataset", local_dir=out_dir)


def _download(url: str, dest: Path):
    print(f"  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_from_dead_sackmann_repo(start_year: int, end_year: int, out_dir: Path):
    """Legacy path, kept in case the original repo ever comes back.

    Confirmed 404 as of this audit -- don't use this by default.
    """
    print(
        "WARNING: JeffSackmann/tennis_atp returned 404 when this was last "
        "checked. Trying anyway, but you almost certainly want "
        "--source huggingface instead.",
        file=sys.stderr,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        _download(
            f"{DEAD_SACKMANN_RAW_BASE}/atp_matches_{year}.csv",
            out_dir / f"atp_matches_{year}.csv",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=["huggingface", "dead-sackmann-repo"],
        default="huggingface",
        help="Where to pull from. 'dead-sackmann-repo' is a long shot kept for posterity.",
    )
    parser.add_argument("--start-year", type=int, default=2000, help="dead-sackmann-repo only")
    parser.add_argument("--end-year", type=int, default=2024, help="dead-sackmann-repo only")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.source == "huggingface":
        download_from_huggingface_mirror(out_dir)
    else:
        download_from_dead_sackmann_repo(args.start_year, args.end_year, out_dir)

    print(
        "\nDone. Column layout should match the bundled "
        "atp_matches_till_2022.csv (see src/data.load_matches); adjust "
        "if the mirror's schema drifted. For closing odds, see "
        "scripts/download_odds.py."
    )


if __name__ == "__main__":
    sys.exit(main())
