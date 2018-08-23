from elo_run.run_model import run_model_one_season
from models import MatchPredictions
from models import engine, Base

MatchPredictions.__table__.drop(engine)
Base.metadata.create_all(engine)

# First relevant season is 2003
season = 2003

# Run all seasons
while season < 2018:
    df = run_model_one_season(season)

    df.to_sql(con=engine, index=False, name=MatchPredictions.__tablename__, if_exists='append')

    season += 1