from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
import os

DATABASE_URL = os.environ['DATABASE_URL']

Base = declarative_base()
engine = create_engine(DATABASE_URL)