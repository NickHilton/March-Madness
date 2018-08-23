def update_function(pred_proba, result_p_value, rating, K):
    """
    Given shape parameters, predicted probability and result p value, get updated rating

    :param pred_proba: (float) in [0,1]
    :param result_p_value: (float) in [0,1]
    :param rating: (float)
    :param K: (float) shpae parameter for ELO system determining how much team ratings move after games
    :return:
    """

    new_rating = rating + K*(result_p_value - pred_proba)

    return new_rating


