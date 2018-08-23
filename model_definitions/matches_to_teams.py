from sqlalchemy import Column, Integer, Table

from model_definitions import Base

matches_to_teams = Table('matches_to_teams', Base.metadata,
                         Column('TeamID', Integer, primary_key=True),
                         Column('id', Integer, primary_key=True),
                         Column('mdid', Integer,)
                         )
