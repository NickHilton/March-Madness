import pandas as pd
from sqlalchemy.orm import sessionmaker

from models import engine, Match


def _calculate_distribution(results: pd.DataFrame) -> pd.Series:
    """
    Calculate cumulative density function of all results to use as a method of
    evaluating the strength
    of a given win

    :param results: (pd.DataFrame) df of results
    :return: (pd.Series) cumulative density function of scores
    """
    delta_counts = results.groupby(by="Delta").mdid.count()
    total = delta_counts.sum()
    densityf = delta_counts / total
    cumdf = densityf.cumsum()
    return cumdf


def _response_function(result: int, distribution: pd.Series) -> float:
    """
    Get p value of a given result from a given distribution

    :param result: (int) score difference
    :param distribution: (pd.Series) cumulative density function
    :return: (float) in [0,1]
    """
    p_val = distribution.get(result, None)
    if p_val:
        return p_val
    if result < distribution.index[0]:
        return 0
    else:
        return _response_function(result - 1, distribution)


def _collect_and_calculate() -> tuple:
    """
    Collect results and filter by Home, Away and Neutral

    :return: (tuple(pd.DataFrame)) of home, away and neutral results
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    matches = (
        session.query(Match.Delta, Match.WLoc, Match.mdid)
        .filter(Match.Season >= 2003)
        .all()
    )
    df = pd.DataFrame(matches)

    neutrals = pd.concat(
        [
            (df.loc[df["WLoc"] == "N", ["Delta", "mdid"]]),
            (-df.loc[df["WLoc"] == "N", ["Delta", "mdid"]]),
        ]
    )

    homes = pd.concat(
        [
            (df.loc[df["WLoc"] == "H", ["Delta", "mdid"]]),
            (-df.loc[df["WLoc"] == "A", ["Delta", "mdid"]]),
        ]
    )

    aways = -homes

    return homes, neutrals, aways


home_results, neutral_results, away_results = _collect_and_calculate()

home_distribution = _calculate_distribution(home_results)
away_distribution = _calculate_distribution(away_results)
neutral_distribution = _calculate_distribution(neutral_results)


def home_response(result):
    return _response_function(result, home_distribution)


def away_response(result):
    return _response_function(result, away_distribution)


def neutral_response(result):
    return _response_function(result, neutral_distribution)
