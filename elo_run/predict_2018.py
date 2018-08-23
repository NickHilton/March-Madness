from elo_run.run_model import run_model_one_season
from models import MatchPredictions
from models import engine, Base



MatchPredictions.__table__.drop(engine)
Base.metadata.create_all(engine)

season = 2003
while season < 2019:
    df = run_model_one_season(season)

    df.to_sql(con=engine, index=False, name=MatchPredictions.__tablename__, if_exists='append')

    season += 1