from sqlalchemy import Column, Integer, ForeignKey, VARCHAR
from sqlalchemy.orm import relationship

from model_definitions import Base
from model_definitions.team import Team


class Seed(Base):
    """
    Seed information
    """
    __tablename__ = "seeds"
    TeamID = Column(Integer, ForeignKey(Team.TeamID), primary_key=True, nullable=False)
    Season = Column(Integer, primary_key=True)
    Seed = Column(Integer)
    SeedSlot = Column(VARCHAR(4))

    team = relationship("Team", foreign_keys=[TeamID])
