import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd
import tqdm

from elo_run.evaluation import evaluate_by_season
from elo_run.link import predict
from elo_run.link_functions import *
from elo_run.response_function import home_response, away_response, neutral_response
from elo_run.run_model import run_model_one_season
from elo_run.elo import ELO
from elo_run.update_rating import update_function
from models import EvaluationRecord
from models import engine, Base, SEASON, SEASON_START

# Response functions to assign a score to the severity of a win
response_fns = {"H": home_response, "A": away_response, "N": neutral_response}

# List of parameters to try
# k_list = [25, 40, 45, 80, 150]
k_list = [25, 40, 80, 150, 300]
# seed_list = [-35, -40, -70, -150]
seed_list = [-5, -20, -35, -40]
d_list = [600]
function_list = ["N", "B", "L"]
link_function_list = {"N": normal_link, "B": bi_logistic_link, "L": logistic_link}
# FGP_list = [1200, 1600, 1800, 2500]
FGP_list = [1200, 2500, 5000]
# R_list = [15, 20, 25, 30]
R_list = [20, 100, 400]
# FGP3_list = [-10, 0, 50, 500, 1000]
FGP3_list = [0, 10, 1000, 5000]
rating_list = [0.3, 1, 5]

DEFAULT_SHAPE_PARAM = 600  # of ELO update func


def set_default_params() -> tuple:
    """
    Set default params

    :return: tuple of varying params
    """
    k = 40
    seed = -100  # Lower seeds are better
    function_code = "N"
    link_function = link_function_list[function_code]
    fgp = 100
    fgp3 = 100
    r = 20
    rating = 1

    return rating, k, seed, function_code, link_function, fgp, fgp3, r


def set_up_elo_model(
    k: int, seed: int, link_function: callable, fgp: float, fgp3: float, r: float, rating: float
) -> ELO:
    """
    Set up an elo system and model with given params

    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param link_function: (func) Function to take team rating difference and spread parameter
                and return a probability
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :param rating: (float) rating weighting
    :return: (ELO) elo class with chosen link funcs, response funcs and param weights
    """
    model_params = {
        "rating": rating,
        "seed": seed,
        "FGP": fgp,
        "R": r,
        "FGP3": fgp3,
        "standard_deviation": 600,
        "link": link_function,
    }

    # Initialise ELO class
    elo = ELO(
        link_function=predict,
        response_functions=response_fns,
        update_function=update_function,
        model_params=model_params,
        K=k,
    )

    return elo


def run_system(elo: ELO, end_season: int = SEASON - 1) -> pd.DataFrame:
    """
    Run Elo system over time, from the start season to the season specified

    :param elo: (ELO) elo class with chosen link funcs, response funcs and param weights
    :param end_season: (int) final season to predict
    :return: (pd.DataFrame) of match predictions, with columns
        [
            "match_id", "Season", "Stage", "WTeamRatingBefore", "LTeamRatingBefore",
            "WTeamRatingAfter", "LTeamRatingAfter", "WTeamID", "LTeamID",
            "PredProbWTeam", "ResultPValue",
        ]
    """
    match_predictions = pd.DataFrame()
    # Initial ratings
    rating_seeds = None
    season = SEASON_START

    while season <= end_season:
        df = run_model_one_season(season, elo_model=elo, rating_seeds=rating_seeds)
        match_predictions = pd.concat([match_predictions, df])

        # Get most recent rating for each team
        last_rating_df = pd.DataFrame()
        # The last match for each team could have been a win or a loss
        for wl in ["W", "L"]:
            # Get last match and rating
            wldf = (
                df.groupby(by=f"{wl}TeamID")
                .tail(1)
                .loc[:, ["match_id", f"{wl}TeamID", f"{wl}TeamRatingAfter"]]
                .rename(
                    columns={f"{wl}TeamID": "Team", f"{wl}TeamRatingAfter": "Rating"}
                )
            )
            last_rating_df = pd.concat([last_rating_df, wldf])

        # Sort by match id and take the most recent rating to use as initial ratings
        # for the next season
        rating_seeds = (
            last_rating_df.sort_values(by="match_id")
            .groupby(by="Team")
            .tail(1)
            .set_index("Team")
            .Rating.to_dict()
        )

        season += 1

    return match_predictions


def save_evaluation(
    rating: float,
    k: int,
    seed: int,
    function_code: str,
    fgp: float,
    fgp3: float,
    r: float,
    match_predictions: pd.DataFrame,
):
    """
    Save evaluations for further analysis

    :param rating: (float) rating param weighting
    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param function_code: (str) Function code for link function
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :param match_predictions: (pd.DataFrame) with columns ['PredProbWTeam', 'Stage']
    :return: (None) -> Saves evaluations to the db
    """
    Base.metadata.create_all(engine)

    results = []

    season = SEASON_START
    while season < SEASON:
        # Get the seasons predictions
        season_predictions = match_predictions.loc[
            match_predictions["Season"] == season, ["PredProbWTeam", "Stage"]
        ].copy()

        # Get evaluation metrics for this season from predictions
        tournament_loss, correct_predictions = evaluate_by_season(
            season_predictions
        )

        result_strings = [
            str(x) for x in (rating, k, seed, function_code, fgp, r, fgp3, season)
        ]
        evaluation_id = "_".join(result_strings) + f"_{DEFAULT_SHAPE_PARAM}"

        results.append(
            (
                evaluation_id,
                rating,
                k,
                seed,
                function_code,
                fgp,
                r,
                fgp3,
                season,
                tournament_loss,
                correct_predictions,
                DEFAULT_SHAPE_PARAM,
            )
        )

        season += 1

    # Create dataframe of results
    df = pd.DataFrame(
        results,
        columns=[
            "id",
            "rating",
            "k",
            "seed",
            "link",
            "FGP",
            "R",
            "FGP3",
            "season",
            "tournament_loss",
            "correct_predictions",
            "d",
        ],
    )

    # Save to the db
    df.to_sql(
        con=engine, index=False, name=EvaluationRecord.__tablename__, if_exists="append"
    )


def run_full_evaluation(
    k: int, seed: int, function_code: str, fgp: float, fgp3: float, r: float, rating: float
) -> None:
    """
    Run whole process for given params

    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param function_code: (str) Function code for link function
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :param rating: (float) rating param weighting
    :return: (None) -> saves evaluations to the
    """
    link_function = link_function_list[function_code]
    elo = set_up_elo_model(
        k=k, seed=seed, link_function=link_function, fgp=fgp, fgp3=fgp3, r=r,
        rating=rating
    )
    match_predictions = run_system(elo)

    save_evaluation(
        rating=rating,
        k=k,
        seed=seed,
        function_code=function_code,
        fgp=fgp,
        fgp3=fgp3,
        r=r,
        match_predictions=match_predictions,
    )


def evaluate_args(args):
    """
    Helper to enable `run_full_evaluation` to be used with `imap`
    :param args: to pass to `run_full_evaluation`
    :return:
    """
    return run_full_evaluation(*args)


# Run the grid search of evaluations
if __name__ == "__main__":
    # Get full param spave
    lists = [k_list, seed_list, function_list, FGP_list, FGP3_list, R_list, rating_list]
    param_space = list(itertools.product(*lists))

    # Use all bar 1 CPU
    pool = Pool(processes=cpu_count() - 1)

    # Track progress and run evaluations
    for _ in tqdm.tqdm(
        pool.imap_unordered(evaluate_args, param_space), total=len(param_space)
    ):
        pass
