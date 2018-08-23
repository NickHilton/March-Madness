class ELO:
    """
    ELO class for implementing the elo system most famously used in chess

    """

    def __init__(self, link_function, response_functions,
                 update_function, model_params, K):

        self.link = link_function
        self.response_functions = response_functions
        self.update = update_function
        self.model_params = model_params
        self.K = K

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

        x_a_updated = self.update(prob_a_wins, result_likelihood, x_a, self.K)

        return x_a_updated

    def response(self, y, location):
        """
        Get p value of a result

        :param y: (int) score difference of team a vs team b. Note that losses for team 1 are negative
        :param location: (str) of team a in ['H', 'A', 'N']
        :return: (float) p value of result
        """
        response = self.response_functions[location](y)

        return response

    def predict(self, x_a, x_b):
        """
        Predict probability of team a beating team b

        :param x_a: (dict(keys = [rating, seed, FGP, R, FGP3])
        :param x_b: (dict(keys = [rating, seed, FGP, R, FGP3])
        :return: (float) in [0,1] predicted probability of team a winning
        """

        return self.link(x_a, x_b, self.model_params)
