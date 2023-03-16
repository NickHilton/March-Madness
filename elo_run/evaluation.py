from math import log
from typing import Tuple
import pandas as pd


def LogLoss(pred_prob_winner: float) -> float:
    """
    Log loss evaluation metric.
    Used to evaluate success by inputting the predicted probability of a
    winner of a match

    :param pred_prob_winner: (float) in range [0,1]
    :return: (float) log loss from the prediction of the winner
    """
    return -1 / 2 * (log(pred_prob_winner + 0.001))  # Add padding to avoid log(0) error


def brier_loss(pred_prob_winner: float) -> float:
    """
    Brier loss evaluation metric:
    :param pred_prob_winner: (float) in range [0,1]
    :return: (float) Brier loss from the prediction of the winner
    """
    return (1 - pred_prob_winner) ** 2


def evaluate_by_season(season_predictions: pd.DataFrame) -> Tuple[float, float]:
    """
    Evaluate success of model for a given season having populated the database
    with predictions of each match winner

    Evaluates solely on tournament games

    :param season_predictions: (pd.DataFrame) season predictions
    :return: (float, float) Average log loss for the tournament matches, percentage of
        winners picked
    """

    # Get brier loss for each prediction
    season_predictions["loss"] = season_predictions.PredProbWTeam.apply(brier_loss)

    # Get correct prediction percentages
    season_predictions["correct_prediction"] = (
        season_predictions.PredProbWTeam > 0.5
    ).astype(int)

    # Aggregate
    tournament_loss = season_predictions.query("Stage == 'T'").loss.mean()
    correct_predictions = season_predictions.query(
        "Stage == 'T'"
    ).correct_prediction.mean()

    return tournament_loss, correct_predictions
