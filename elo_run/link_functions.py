from scipy.stats import norm

def normal_link(theta, d):
    """

    :param theta: (float) z value to investigate
    :param d: (float) standard deviation of normal cdf to use
    :return:
    """
    return norm.cdf(theta / d)


def logistic_link(theta, d):
    return 1 / (1 + 10 ** (-theta / d))


def bi_logistic_link(theta, d):
    low_dist = 1 / (1 + 10 ** (-(theta - 2 * d) / d))
    high_dist = 1 / (1 + 10 ** (-(theta + 2 * d) / d))

    return 0.5 * (high_dist + low_dist)

