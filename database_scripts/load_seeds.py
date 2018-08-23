import pandas as pd
from models import Base, engine

def load_seeds():
    """
    Load seeds into database

    :return: (None)
    """
    # Add Teams
    Base.metadata.create_all(engine)
    file_name = '../data/NCAATourneySeeds_updated.csv'
    df = pd.read_csv(file_name)
    df.Seed = df.Seed.apply(lambda x: int(x[1:3]))

    df.to_sql(con=engine, index=False, name='seeds', if_exists='replace')

load_seeds()
