import sqlite3

import pandas as pd

import os

DATABASE_URL = os.environ["DATABASE_URL"]
DATABASE = DATABASE_URL.split('sqlite:///')[1]


def sqlite_to_df(query, db_file=DATABASE):
    conn = sqlite3.connect(db_file)

    cur = conn.cursor()

    results = cur.execute(query).fetchall()

    names = list(map(lambda x: x[0], cur.description))

    df = pd.DataFrame(results, columns=names)

    return df
