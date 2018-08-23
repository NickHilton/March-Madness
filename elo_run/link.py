from elo_run.link_functions import *
import numpy as np

model_params_default = {
    'rating': 1,
    'seed': 10,
    'FGP': 100,
    'R': 10,
    'FGP3': 100,
    'standard_devation': 600,
    'link': normal_link
}


def predict(team_1, team_2, model_params=model_params_default):
    """
    Predicts the probablity that team_1 beats team_2 for a given set of model params
    :param team_1: (dict(keys = [rating, seed, FGP, R, FGP3]))
    :param team_2: (dict(keys = [rating, seed, FGP, R, FGP3]))
    :param model_params:
    :return:
    """

    use_seeds = True if team_1['seed'] and team_2['seed'] else False

    params = ['rating', 'FGP', 'R', 'FGP3']
    num_params = 4

    if use_seeds:
        params.append('seed')
        num_params = 5

        if team_1['seed'] - team_2['seed'] >= 15:
            return 0.01
        if team_1['seed'] - team_2['seed'] <= -15:
            return 0.99

    t1_vec = np.empty(num_params)
    t2_vec = np.empty(num_params)
    param_vec = np.empty(num_params)

    for n, param in enumerate(params):
        t1_vec[n] = team_1[param]
        t2_vec[n] = team_2[param]
        param_vec[n] = model_params[param]

    theta = param_vec.dot(t1_vec - t2_vec)

    d = model_params['standard_devation']

    pred_prob = model_params['link'](theta, d)

    return pred_prob



