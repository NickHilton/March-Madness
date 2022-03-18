from scipy.stats import norm


def normal_link(theta: float, d: float) -> float:
    """
    Normal cdf function to use as a link from two team ratings to a predicted probability

    :param theta: (float) z value to investigate
    :param d: (float) standard deviation of normal cdf to use
    :return: (float) in [0,1]
    """
    return norm.cdf(theta / d)


def logistic_link(theta: float, d: float) -> float:
    """
    Logistic function to use as a link from two team ratings to a predicted probability

    :param theta: (float) value to investigate
    :param d: (float) spread parameter
    :return: (float)
    """
    return 1 / (1 + 10 ** (-theta / d))


def bi_logistic_link(theta: float, d: float) -> float:
    """
    Bi modal logistic function to use as a link from two team ratings to a
    predicted probability

    :param theta: (float) value to investigate
    :param d: (float) spread parameter
    :return: (float) in [0,1]
    """

    low_dist = 1 / (1 + 10 ** (-(theta - 2 * d) / d))
    high_dist = 1 / (1 + 10 ** (-(theta + 2 * d) / d))

    return 0.5 * (high_dist + low_dist)
