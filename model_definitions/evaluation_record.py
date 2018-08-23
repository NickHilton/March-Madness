from sqlalchemy import Column, Integer,VARCHAR, Float, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship

from model_definitions import Base
from model_definitions.match import Match


class EvaluationRecord(Base):
    __tablename__ = 'evaluations'
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


