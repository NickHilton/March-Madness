import pandas as pd
from sqlalchemy.orm import sessionmaker

from models import engine, Match


def _calculate_distribution(results):
    delta_counts = results.groupby(by='Delta').mdid.count()
    total = delta_counts.sum()
    densityf = delta_counts / total
    cumdf = densityf.cumsum()
    return cumdf


def _response_function(result, distribution):
    p_val = distribution.get(result, None)
    if p_val:
        return p_val
    if result < distribution.index[0]:
        return 0
    else:
        return _response_function(result - 1, distribution)


def _collect_and_calculate():
    Session = sessionmaker(bind=engine)
    session = Session()
    matches = session.query(Match.Delta, Match.WLoc, Match.mdid).filter(Match.Season >= 2003).all()
    df = pd.DataFrame(matches)

    neutrals = (df.loc[df['WLoc'] == 'N', ['Delta', 'mdid']]
                ).append(-df.loc[df['WLoc'] == 'N', ['Delta', 'mdid']])

    homes = (df.loc[df['WLoc'] == 'H', ['Delta', 'mdid']]
             ).append(-df.loc[df['WLoc'] == 'A', ['Delta', 'mdid']])

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
