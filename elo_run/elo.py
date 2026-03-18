class ELO:
    """
    ELO class for implementing the elo system most famously used in chess

    """

    def __init__(
        self, link_function, response_functions, update_function, model_params, K, alpha=0.0, decay=1.0
    ):

        self.link = link_function
        self.response_functions = response_functions
        self.update = update_function
        self.model_params = model_params
        self.K = K
        self.alpha = alpha
        self.decay = decay

    def update_rating(self, x_a, x_b, y, location):
        """
        Given two initial team ratings and a result, update the first teams rating

        :param x_a: (dict(keys = [rating, seed, FGP, R, FGP3])
        :param x_b: (dict(keys = [rating, seed, FGP, R, FGP3])
        :param y: (int) result score difference
        :param location (str)
        :return: (float) new team rating for team a
        """

        prob_a_wins = self.predict(x_a, x_b)

        result_likelihood = self.response(y, location)

        x_a_updated = self.update(prob_a_wins, result_likelihood, x_a["rating"], self.K)

        return x_a_updated

    def response(self, y, location):
        """
        Get p value of a result

        :param y: (int) score difference of team a vs team b.
            Note that losses for team 1 are negative
        :param location: (str) of team a in ['H', 'A', 'N']
        :return: (float) p value of result
        """
        margin_response = self.response_functions[location](y)
        if self.alpha == 0.0:
            return margin_response
        win_indicator = 1.0 if y > 0 else (0.5 if y == 0 else 0.0)
        return self.alpha * win_indicator + (1 - self.alpha) * margin_response

    def predict(self, x_a, x_b):
        """
        Predict probability of team a beating team b

        :param x_a: (dict(keys = [rating, seed, FGP, R, FGP3])
        :param x_b: (dict(keys = [rating, seed, FGP, R, FGP3])
        :return: (float) in [0,1] predicted probability of team a winning
        """

        return self.link(x_a, x_b, self.model_params)
