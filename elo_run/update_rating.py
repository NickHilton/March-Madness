def update_function(pred_proba, result_p_value, rating, K):

    new_rating = rating + K*(result_p_value - pred_proba)

    return new_rating


