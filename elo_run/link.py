from typing import Optional

import numpy as np

from elo_run.link_functions import *

# Default model parameters
model_params_default = {
    "rating": 1,
    "seed": 10,
    "FGP": 100,
    "R": 10,
    "FGP3": 100,
    "standard_deviation": 600,
    "link": normal_link,
}


def predict(team_1: dict, team_2: dict, model_params: Optional[dict] = None) -> float:
    """
    Predicts the probability that team_1 beats team_2 for a given set of model params

    :param team_1: (dict(keys = [rating, seed, FGP, R, FGP3]))
    :param team_2: (dict(keys = [rating, seed, FGP, R, FGP3]))
    :param model_params: (dict) of model parameters and their respective values
    :return: (float)
    """
    if not model_params:
        model_params = model_params_default

    use_seeds = True if (team_1["seed"] and team_2["seed"]) else False

    # Team rating, Field goal percentage, average rebounds,
    # 3 point field goal percentage
    params = ["rating", "FGP", "R", "FGP3"]

    if use_seeds:
        # Add in seed information
        params.append("seed")
        #
        # # For a 1 vs 16 seed give the 1 seed a 99% predicted probability
        # if team_1["seed"] - team_2["seed"] >= 15:
        #     return 0.01
        # if team_1["seed"] - team_2["seed"] <= -15:
        #     return 0.99

    num_params = len(params)

    # Initialise np vectors
    t1_vec = np.empty(num_params)
    t2_vec = np.empty(num_params)
    param_vec = np.empty(num_params)

    for n, param in enumerate(params):
        t1_vec[n] = team_1[param]
        t2_vec[n] = team_2[param]
        param_vec[n] = model_params[param]

    # Value to input into link function
    theta = param_vec.dot(t1_vec - t2_vec)
    theta = max(min(theta, 1e300), -1e300)

    d = model_params["standard_deviation"]

    pred_prob = model_params["link"](theta, d)

    return pred_prob
