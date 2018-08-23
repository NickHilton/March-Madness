from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()
engine = create_engine('sqlite:////Users/nickhilton/Documents/turtle/databases/turtle.db')