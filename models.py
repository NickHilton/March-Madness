from model_definitions import Base, engine
from model_definitions.match import Match
from model_definitions.team import Team
from model_definitions.seed import Seed
from model_definitions.matches_to_teams import matches_to_teams
from model_definitions.match_predictions import MatchPredictions
from model_definitions.evaluation_record import EvaluationRecord

Base.metadata.create_all(bind=engine)