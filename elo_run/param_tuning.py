import pandas as pd

from elo_run.evaluation import evaluate_by_season
from elo_run.link import predict
from elo_run.link_functions import *
from elo_run.response_function import home_response, away_response, neutral_response
from elo_run.run_model import run_model_one_season
from elo_run.set_up_elo import ELO
from elo_run.update_rating import update_function
from models import EvaluationRecord
from models import engine, Base, MatchPredictions

# Response functions to assign a score to the severity of a win
response_fns = {
    'H': home_response,
    'A': away_response,
    'N': neutral_response
}

# List of parameters to try
# These have been updated having already completed some runs
k_list = [36, 41, 447]
seed_list = [-7, -11, -22, -32, -50]
d_list = [600]
function_list = ['N', 'B', 'L']
link_function_list = {
    'N': normal_link,
    'B': bi_logistic_link,
    'L': logistic_link
}
FGP_list = [0, 500, 1000, 2000, 10000]
R_list = [15, 20, 25, 30]
FGP3_list = [0, 500, 1000, 2000, 10000]


def set_default_params():
    """
    Set default params

    :return: tuple of varying params
    """
    k = 40
    seed = -100  # Lower seeds are better
    function_code = 'N'
    link_function = link_function_list[function_code]
    fgp = 100
    fgp3 = 100
    r = 20
    rating = 1

    return rating, k, seed, function_code, link_function, fgp, fgp3, r


def set_up_elo_model(k, seed, link_function, fgp, fgp3, r):
    """
    Set up an elo system and model with given params

    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param link_function: (func) Function to take team rating difference and spread parameter
                and return a probability
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :return: (ELO) elo class with chosen link funcs, response funcs and param weights
    """
    model_params = {
        'rating': 1,
        'seed': seed,
        'FGP': fgp,
        'R': r,
        'FGP3': fgp3,
        'standard_devation': 600,
        'link': link_function
    }

    # Initialise ELO class
    elo = ELO(link_function=predict,
              response_functions=response_fns,
              update_function=update_function,
              model_params=model_params,
              K=k)

    return elo


def run_system(elo):
    """
    Run Elo system over time

    :param elo: (ELO) elo class with chosen link funcs, response funcs and param weights
    :return: (None)
    """
    MatchPredictions.__table__.drop(engine)
    Base.metadata.create_all(engine)

    season = 2003
    while season < 2018:
        df = run_model_one_season(season, elo_model=elo)

        df.to_sql(con=engine, index=False, name=MatchPredictions.__tablename__, if_exists='append')

        season += 1


def save_evaluation(rating, k, seed, function_code, fgp, fgp3, r):
    """
    Save evaluations for further analysis

    :param rating: (float) rating param weighting
    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param function_code: (str) Function code for link function
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :return: (None)
    """
    Base.metadata.create_all(engine)
    results = []
    season = 2003
    while season < 2018:
        tournament_log_loss, correct_predictions = evaluate_by_season((season))

        result_strings = [str(x) for x in (rating, k, seed, function_code, fgp, r, fgp3, season)]
        id = '_'.join(result_strings) + '_600'

        results.append(
            (id, rating, k, seed, function_code, fgp, r, fgp3, season, tournament_log_loss, correct_predictions, 600))

        season += 1

    df = pd.DataFrame(results, columns=['id', 'rating', 'k', 'seed', 'link', 'FGP', 'R', 'FGP3',
                                        'season', 'tournament_log_loss', 'correct_predictions', 'd'])

    df.to_sql(con=engine, index=False, name=EvaluationRecord.__tablename__, if_exists='append')


def run_full_evaluation(k, seed, function_code, link_function, fgp, fgp3, r):
    """
    Run whole process for given params

    :param rating: (float) rating param weighting
    :param k: (int) k value in elo system - how far team ratings will move after one match
    :param seed: (int) seed weighting
    :param function_code: (str) Function code for link function
    :param fgp: (float) field goal percentage weighting
    :param fgp3: (float) 3pt field goal percentage weighting
    :param r: (float) rebound weighting
    :return: (None)
    """
    elo = set_up_elo_model(k, seed, link_function, fgp, fgp3, r)
    run_system(elo)
    save_evaluation(1, k, seed, function_code, fgp, fgp3, r)


# Run initial process
rating, k, seed, function_code, link_function, fgp, fgp3, r = set_default_params()

run_full_evaluation(k, seed, function_code, link_function, fgp, fgp3, r)


# Test different params
for seed_test in seed_list:
    if seed_test != seed:
        run_full_evaluation(k, seed_test, function_code, link_function, fgp, fgp3, r)

for k_test in k_list:
    if k_test != k:
        run_full_evaluation(k_test, seed, function_code, link_function, fgp, fgp3, r)

for fn_code_test in function_list:
    if fn_code_test != function_code:
        link_test = link_function_list[fn_code_test]
        run_full_evaluation(k, seed, fn_code_test, link_test, fgp, fgp3, r)

for fgp_test in FGP_list:
    if fgp_test != fgp:
        run_full_evaluation(k, seed, function_code, link_function, fgp_test, fgp3, r)

for r_test in R_list:
    if r_test != r:
        run_full_evaluation(k, seed, function_code, link_function, fgp, fgp3, r_test)

for fgp3_test in FGP3_list:
    if fgp3_test != fgp3:
        run_full_evaluation(k, seed, function_code, link_function, fgp, fgp3_test, r)
