"""
Load and cache Massey Ordinals composite rankings.

Uses day-133 (final pre-tournament) rankings from reliable systems,
averaged into a single composite rank per team per season.
"""
import json
import csv
import os
from collections import defaultdict

# Systems with 15 seasons of coverage (2010-2025), 340+ teams ranked
SYSTEMS = {"POM", "MOR", "COL", "DOK", "WIL", "MAS", "KPK", "PGH", "BIH", "WLK"}

# Module-level cache: {season: {team_id: composite_rank}}
_massey_cache = {}
_massey_loaded = False


def _load_massey():
    """Load composite ranks, using a JSON cache for speed."""
    global _massey_loaded
    if _massey_loaded:
        return

    data_path = os.environ.get("DATA_PATH", "data_male")
    data_prefix = os.environ.get("DATA_PREFIX", "M")
    csv_path = f"{data_path}/{data_prefix}MasseyOrdinals.csv"
    cache_path = f"{data_path}/{data_prefix}MasseyComposite.json"

    if not os.path.exists(csv_path):
        _massey_loaded = True
        return

    # Try fast JSON cache first
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(csv_path):
        with open(cache_path) as f:
            data = json.load(f)
        for season_str, teams in data.items():
            _massey_cache[int(season_str)] = {int(tid): rank for tid, rank in teams.items()}
        _massey_loaded = True
        return

    # Build from CSV (slow, ~6s for 5.8M rows)
    # Columns: Season(0), RankingDayNum(1), SystemName(2), TeamID(3), OrdinalRank(4)
    rank_sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row[1] == "133" and row[2] in SYSTEMS:
                season = int(row[0])
                team_id = int(row[3])
                rank = int(row[4])
                entry = rank_sums[season][team_id]
                entry[0] += rank
                entry[1] += 1

    for season, teams in rank_sums.items():
        _massey_cache[season] = {
            tid: total / count for tid, (total, count) in teams.items()
        }

    # Write JSON cache for next time
    cache_data = {
        str(s): {str(tid): rank for tid, rank in teams.items()}
        for s, teams in _massey_cache.items()
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)

    _massey_loaded = True


def get_massey_ranks(season: int) -> dict:
    """
    Get composite Massey rank for all teams in a given season.

    :param season: (int) season year
    :return: dict of {team_id: composite_rank} (lower = better)
    """
    _load_massey()
    return _massey_cache.get(season, {})
