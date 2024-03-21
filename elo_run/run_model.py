from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, and_
from sqlalchemy.orm import sessionmaker

from models import engine, Seed, Match, Team, MatchPredictions
from .MatchStack import MatchStack
from .link import predict
from .link_functions import *
from .response_function import home_response, away_response, neutral_response
from .elo import ELO
from .update_rating import update_function

response_fns = {"H": home_response, "A": away_response, "N": neutral_response}

model_params_default = {
    "rating": 1,
    "seed": -10,
    "FGP": 100,
    "R": 10,
    "FGP3": 100,
    "standard_deviation": 600,
    "link": normal_link,
}


elo = ELO(
    link_function=predict,
    response_functions=response_fns,
    update_function=update_function,
    model_params=model_params_default,
    K=40,
)

DEFAULT_RATING = 1500


def season_teams(season):
    """
    Get team_ids for a given season

    :param season: (int) season
    :return: (list(int))
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    _ = session.query(Team)

    team_ids = (
        session.query(Team.TeamID)
        .join(Match.teams)
        .filter(Match.Season == season)
        .group_by(Team.TeamID)
        .all()
    )

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
    team_to_seed = dict(
        session.query(Seed.TeamID, Seed.Seed).filter(Seed.Season == Season).all()
    )
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

    q = (
        session.query(Team.TeamID, func.max(Match.mdid).label("mdid"))
        .join(Match.teams)
        .filter(Match.Season == Season - 1)
        .group_by(Team.TeamID)
        .subquery()
    )

    winners = (
        session.query(Match.WTeamID, MatchPredictions.WTeamRatingAfter)
        .join(q, and_(Match.WTeamID == q.c.TeamID, Match.mdid == q.c.mdid))
        .join(MatchPredictions)
        .all()
    )

    losers = (
        session.query(Match.LTeamID, MatchPredictions.LTeamRatingAfter)
        .join(q, and_(Match.LTeamID == q.c.TeamID, Match.mdid == q.c.mdid))
        .join(MatchPredictions)
        .all()
    )

    team_to_rating = dict(winners + losers)

    return team_to_rating


def set_up(Season: int, rating_seeds: Optional[dict] = None):
    """
    Set Get all match and team info

    :param Season: (int) season
    :param rating_seeds: (dict or None) of initial team ratings
    :return:
    """
    team_info = defaultdict(lambda: dict())

    team_ids = season_teams(Season)
    team_to_seed = season_team_seeds(Season)
    if not rating_seeds:
        rating_seeds = dict()

    default_rating = np.mean(list(rating_seeds.values()))
    if np.isnan(default_rating):
        print("No ratings found, using default rating")
        default_rating = DEFAULT_RATING
    # print(f"Default rating: {default_rating}")

    for TeamID in team_ids:
        team_info[TeamID]["seed"] = team_to_seed.get(TeamID, None)
        team_info[TeamID]["rating"] = rating_seeds.get(TeamID, default_rating)

    season_match_stack = MatchStack(Season)

    matches_df = pd.DataFrame(season_match_stack.matches)

    matches_df["WSeed"] = matches_df.WTeamID.map(team_to_seed)
    matches_df["LSeed"] = matches_df.LTeamID.map(team_to_seed)

    return matches_df, team_info


def run_model_one_season(
    season: int, elo_model: ELO = elo, rating_seeds: Optional[dict] = None
) -> pd.DataFrame:
    """
    Run the ELO model for one season, returning a dataframe of all predictions for each
    match with columns:
        [
            "match_id", "Season", "Stage", "WTeamRatingBefore", "LTeamRatingBefore",
            "WTeamRatingAfter", "LTeamRatingAfter", "WTeamID", "LTeamID",
            "PredProbWTeam", "ResultPValue",
        ]
    :param season: (int) season to run
    :param elo_model: (ELO) the underlying ELO model
    :param rating_seeds: (dict or None) of initial ELO ratings to use
    :return: (pd.DataFrame) of predictions and ratings for each match
    """

    df, teams = set_up(season, rating_seeds=rating_seeds)

    results = []
    for _, row in df.iterrows():
        wteam_id = row.WTeamID
        lteam_id = row.LTeamID
        wteam = {}
        lteam = {}

        if ~np.isnan(row.WSeed):
            wteam["seed"] = row.WSeed
        else:
            wteam["seed"] = None

        if ~np.isnan(row.LSeed):
            lteam["seed"] = row.LSeed
        else:
            lteam["seed"] = None

        wteam["rating"] = teams[wteam_id]["rating"]
        lteam["rating"] = teams[lteam_id]["rating"]

        wteam["FGP"] = row.WFGP_avg
        wteam["R"] = row.WR_avg
        wteam["FGP3"] = row.WFGP3_avg
        lteam["FGP"] = row.LFGP_avg
        lteam["R"] = row.LR_avg
        lteam["FGP3"] = row.LFGP3_avg

        w_old = wteam["rating"]
        l_old = lteam["rating"]

        pred_prob = elo_model.predict(wteam, lteam)
        result_likelihood = elo_model.response(row.Delta, row.WLoc)

        w_new = elo_model.update(pred_prob, result_likelihood, w_old, elo.K)
        teams[row.WTeamID]["rating"] = w_new

        l_new = elo_model.update(
            1 - pred_prob, 1 - result_likelihood, l_old, elo_model.K
        )
        teams[row.LTeamID]["rating"] = l_new

        results.append(
            (
                row.id,
                season,
                row.stage,
                w_old,
                l_old,
                w_new,
                l_new,
                wteam_id,
                lteam_id,
                pred_prob,
                result_likelihood,
            )
        )

    res_df = pd.DataFrame(
        results,
        columns=[
            "match_id",
            "Season",
            "Stage",
            "WTeamRatingBefore",
            "LTeamRatingBefore",
            "WTeamRatingAfter",
            "LTeamRatingAfter",
            "WTeamID",
            "LTeamID",
            "PredProbWTeam",
            "ResultPValue",
        ],
    )

    return res_df
