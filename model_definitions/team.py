from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import relationship

from model_definitions import Base
from model_definitions.matches_to_teams import matches_to_teams


class Team(Base):
    __tablename__ = 'teams'
    TeamID = Column(Integer, primary_key=True, nullable=False)
    FirstD1Season = Column(Integer)
    LastD1Season = Column(Integer)
    TeamName = Column(VARCHAR(32))

    matches = relationship('Match',
                           secondary=matches_to_teams,
                           primaryjoin="teams.c.TeamID==matches_to_teams.c.TeamID",
                           secondaryjoin="matches_to_teams.c.id==matches.c.id",
                           collection_class=ordering_list('mdid'),
                           backref='teams'
                           )