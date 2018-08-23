class ELO:

    def __init__(self, link_function, response_functions,
                 update_function, model_params, K):

        self.link = link_function
        self.response_functions = response_functions
        self.update = update_function
        self.model_params = model_params
        self.K = K

    def update_rating(self, x_a, x_b, y, location):
        """

        :param x_a: (np.array)
        :param y: (
        :param x_b: (np.array)
        :return:
        """

        prob_a_wins = self.predict(x_a, x_b)

        result_likelihood = self.response(y, location)

        x_a_updated = self.update(prob_a_wins, result_likelihood, x_a, self.K)

        return x_a_updated

    def response(self, y, location):
        """

        :param y:
        :param location:
        :return:
        """
        response = self.response_functions[location](y)

        return response

    def predict(self, x_a, x_b):
        """

        :param x_a:
        :param x_b:
        :return:
        """

        return self.link(x_a, x_b, self.model_params)
