from sqlalchemy import Column, Integer,VARCHAR, Float, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship

from model_definitions import Base
from model_definitions.match import Match


class MatchPredictions(Base):
    __tablename__ = 'match_predictions'
    match_id = Column(Integer, ForeignKey(Match.id), primary_key=True)
    Season = Column(Integer)
    Stage = Column(VARCHAR(1))

    WTeamRatingBefore = Column(Float)
    LTeamRatingBefore = Column(Float)
    WTeamRatingAfter = Column(Float)
    LTeamRatingAfter = Column(Float)

    PredProbWTeam = Column(Float)
    ResultPValue = Column(Float)

    match = relationship('Match', foreign_keys=[match_id], backref='prediction')
    season = association_proxy('match', 'Season')
    stage = association_proxy('match', 'Stage')

