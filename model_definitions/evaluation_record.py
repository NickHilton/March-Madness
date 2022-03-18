from sqlalchemy import Column, Integer, VARCHAR, Float

from model_definitions import Base


class EvaluationRecord(Base):
    """
    Records results of performing an experiment
    season -> Season of evaluation
    [tournament_log_loss, correct_predictions] -> metrics
    OTHER -> params
    """

    __tablename__ = "evaluations"
    id = Column(VARCHAR(300), primary_key=True, nullable=False)
    rating = Column(Float)
    d = Column(Float)
    k = Column(Integer)
    seed = Column(Float)
    link = Column(VARCHAR(1))
    FGP = Column(Float)
    R = Column(Float)
    FGP3 = Column(Float)
    season = Column(Integer)
    tournament_log_loss = Column(Float)
    correct_predictions = Column(Float)
