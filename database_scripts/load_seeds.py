import pandas as pd
from models import Base, engine
import os
DATA_PATH = os.environ['DATA_PATH']
DATA_PREFIX = os.environ['DATA_PREFIX']

def load_seeds():
    """
    Load seeds into database

    :return: (None)
    """
    # Add Teams
    Base.metadata.create_all(engine)
    file_name = f'{DATA_PATH}/{DATA_PREFIX}NCAATourneySeeds.csv'
    df = pd.read_csv(file_name)
    df.Seed = df.Seed.apply(lambda x: int(x[1:3]))

    df.to_sql(con=engine, index=False, name='seeds', if_exists='replace')

load_seeds()
