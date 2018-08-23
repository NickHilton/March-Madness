from math import log

import pandas as pd
from sqlalchemy.orm import sessionmaker

from models import MatchPredictions, engine


def LogLoss(pred_prob_winner):
    """
    Log loss evaluation metric.
    Used to evaluate success by inputting the predicted probability of a winner of a match

    :param pred_prob_winner: (float) in range [0,1]
    :return: (float)
    """
    return - 1 / 2 * (log(pred_prob_winner))


def evaluate_by_season(Season):
    """
    Evaluate success of model for a given season having populated the database with predictions of each
    match winner

    Evaluates soley on tournament games

    :param Season: (int) season
    :return: (float, float) Average log loss for the tournament matches, percentage of winners picked
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    predictions = session.query(
        MatchPredictions.PredProbWTeam,
        MatchPredictions.Stage
    ).filter(
        MatchPredictions.Season == Season
    ).all()

    df = pd.DataFrame(predictions)
    df['log_loss'] = df.PredProbWTeam.apply(LogLoss)

    df['correct_prediction'] = (df.PredProbWTeam > 0.5).astype(int)

    tournament_log_loss = df.query("Stage == 'T'").log_loss.mean()
    correct_predictions = df.query("Stage == 'T'").correct_prediction.mean()

    return tournament_log_loss, correct_predictions
