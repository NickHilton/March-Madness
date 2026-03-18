"""
Load and cache Massey Ordinals composite rankings.

Uses day-133 (final pre-tournament) rankings from reliable systems,
averaged into a single composite rank per team per season.
"""
import os

import pandas as pd

# Systems with 15 seasons of coverage (2010-2025), 340+ teams ranked
SYSTEMS = ["POM", "MOR", "COL", "DOK", "WIL", "MAS", "KPK", "PGH", "BIH", "WLK"]

# Module-level cache: {season: {team_id: composite_rank}}
_massey_cache = {}
_massey_loaded = False


def _load_massey():
    """Load CSV once and build per-season composite rank lookup."""
    global _massey_loaded
    if _massey_loaded:
        return

    data_path = os.environ.get("DATA_PATH", "data_male")
    data_prefix = os.environ.get("DATA_PREFIX", "M")
    csv_path = f"{data_path}/{data_prefix}MasseyOrdinals.csv"

    if not os.path.exists(csv_path):
        _massey_loaded = True
        return

    df = pd.read_csv(csv_path)

    # Use day 133 (final pre-tournament snapshot)
    day133 = df[df.RankingDayNum == 133]

    # Filter to reliable systems
    day133 = day133[day133.SystemName.isin(SYSTEMS)]

    # Compute average rank per team per season
    composite = day133.groupby(["Season", "TeamID"])["OrdinalRank"].mean()

    for (season, team_id), rank in composite.items():
        if season not in _massey_cache:
            _massey_cache[season] = {}
        _massey_cache[season][team_id] = rank

    _massey_loaded = True


def get_massey_ranks(season: int) -> dict:
    """
    Get composite Massey rank for all teams in a given season.

    :param season: (int) season year
    :return: dict of {team_id: composite_rank} (lower = better)
    """
    _load_massey()
    return _massey_cache.get(season, {})
