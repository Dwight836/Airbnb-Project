import sqlite3
import pandas as pd


def query_reader(query, db_name):
    """ Produces a DataFrame given an SQL Query and database name

    Args :
        query (string) : SQL query specifying what data to pull
        db_name (string) : String specifying which database from which to pull

    Returns :
        df (DataFrame) : Pandas DataFrame containing the desired information

    """
    # Establishes connection and creates a DataFrame
    con = sqlite3.connect(db_name)
    df = pd.read_sql_query(query, con)

    # Drops null values
    df.dropna(inplace=True)

    return df
