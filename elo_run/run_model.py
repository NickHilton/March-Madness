from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import func, and_
from sqlalchemy.orm import sessionmaker

from models import engine, Seed, Match, Team, MatchPredictions
from .MatchStack import MatchStack
from .link import predict
from .link_functions import *
from .response_function import home_response, away_response, neutral_response
from .set_up_elo import ELO
from .update_rating import update_function

response_fns = {
    'H': home_response,
    'A': away_response,
    'N': neutral_response
}

model_params_default = {
    'rating': 1,
    'seed': -10,
    'FGP': 100,
    'R': 10,
    'FGP3': 100,
    'standard_devation': 600,
    'link': normal_link
}


elo = ELO(link_function=predict,
          response_functions=response_fns,
          update_function=update_function,
          model_params=model_params_default,
          K=40)

season = 2003


def season_teams(Season):
    """
    Get team_ids for a given season

    :param Season: (int) season
    :return: (list(int))
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    _ = session.query(Team)

    team_ids = session.query(Team.TeamID).join(Match.teams).filter(Match.Season == Season).group_by(Team.TeamID).all()

    team_ids = [TeamID for TeamID, in team_ids]
    session.close()

    return team_ids


def season_team_seeds(Season):
    """
    Get team seeds in the tournament for a given season

    :param Season: (int) season
    :return: (dict)
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    team_to_seed = dict(session.query(Seed.TeamID, Seed.Seed).filter(Seed.Season == Season).all())
    session.close()
    return team_to_seed


def get_most_recent_ratings(Season):
    """
    Get most recent ratings from previous season

    :param Season: (int) season
    :return: (dict)
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    q = session.query(Team.TeamID, func.max(Match.mdid).label('mdid')).join(
        Match.teams).filter(Match.Season == Season - 1).group_by(Team.TeamID).subquery()

    winners = session.query(Match.WTeamID, MatchPredictions.WTeamRatingAfter).join(
        q, and_(Match.WTeamID == q.c.TeamID, Match.mdid == q.c.mdid)
    ).join(MatchPredictions).all()

    losers = session.query(Match.LTeamID, MatchPredictions.LTeamRatingAfter).join(
        q, and_(Match.LTeamID == q.c.TeamID, Match.mdid == q.c.mdid)
    ).join(MatchPredictions).all()

    team_to_rating = dict(winners + losers)

    return team_to_rating


def set_up(Season):
    """
    Set Get all match and team info

    :param Season: (int) season
    :return:
    """
    team_info = defaultdict(lambda: dict())

    team_ids = season_teams(Season)
    team_to_seed = season_team_seeds(Season)
    team_to_rating = get_most_recent_ratings(Season)

    for TeamID in team_ids:
        team_info[TeamID]['seed'] = team_to_seed.get(TeamID, None)
        team_info[TeamID]['rating'] = team_to_rating.get(TeamID, 1500)

    season_match_stack = MatchStack(Season)

    matches_df = pd.DataFrame(season_match_stack.matches)

    matches_df['WSeed'] = matches_df.WTeamID.map(team_to_seed)
    matches_df['LSeed'] = matches_df.LTeamID.map(team_to_seed)

    return matches_df, team_info


def run_model_one_season(Season, elo_model=elo):
    df, teams = set_up(Season)

    results = []
    for _, row in df.iterrows():
        wteam = {}
        lteam = {}

        if ~np.isnan(row.WSeed):
            wteam['seed'] = row.WSeed
        else:
            wteam['seed'] = None

        if ~np.isnan(row.LSeed):
            lteam['seed'] = row.LSeed
        else:
            lteam['seed'] = None

        wteam['rating'] = teams[row.WTeamID]['rating']
        lteam['rating'] = teams[row.LTeamID]['rating']

        wteam['FGP'] = row.WFGP_avg
        wteam['R'] = row.WR_avg
        wteam['FGP3'] = row.WFGP3_avg
        lteam['FGP'] = row.LFGP_avg
        lteam['R'] = row.LR_avg
        lteam['FGP3'] = row.LFGP3_avg

        w_old = wteam['rating']
        l_old = lteam['rating']

        pred_prob = elo_model.predict(wteam, lteam)
        result_likelihood = elo_model.response(row.Delta, row.WLoc)

        w_new = elo_model.update(pred_prob, result_likelihood, w_old, elo.K)
        teams[row.WTeamID]['rating'] = w_new

        l_new = elo_model.update(1 - pred_prob, 1 - result_likelihood, l_old, elo_model.K)
        teams[row.LTeamID]['rating'] = l_new

        results.append((row.id, Season, row.stage, w_old, l_old, w_new, l_new, pred_prob, result_likelihood))

    res_df = pd.DataFrame(results, columns=['match_id',
                                            'Season',
                                            'Stage',
                                            'WTeamRatingBefore',
                                            'LTeamRatingBefore',
                                            'WTeamRatingAfter',
                                            'LTeamRatingAfter',
                                            'PredProbWTeam',
                                            'ResultPValue'])

    return res_df
