import sqlite3
import pandas as pd
from config_reader import get_db_config

db_path = get_db_config()['db_path']

# build the db
def create_exercise_summary_table(db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS exercise_summary(
                  activity_id                 INT PRIMARY KEY
                , name                      TEXT
                , date                      TEXT
                , type                      TEXT
                , distance                  REAL
                , moving_time               INT
                , elapsed_time              INT
                , total_elevation_gain      REAL
                , elev_high                 REAL
                , elev_low                  REAL
                , start_datetime_utc        TEXT
                , timezone                  TEXT
                , start_latitude            REAL
                , start_longitude           REAL
                , end_latitude              REAL
                , end_longitude             REAL
            );
        """)


def existing_activity_ids(proposed_ids, table_name, db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        #conn.set_trace_callback(print)
        cur.execute("""
            select distinct activity_id from """ +
            table_name +
            """
            where activity_id in ({0});
            """.format(", ".join('?' for _ in proposed_ids)),
            proposed_ids
        )
        existing_ids = cur.fetchall()
        existing_ids = [k[0] for k in existing_ids]

    return [k for k in proposed_ids if k in existing_ids]


def append_exercise_summary(df, db_path=db_path):
    proposed_ids = list(set([int(x) for x in df['activity_id'].to_numpy()]))
    old_ids = existing_activity_ids(proposed_ids, 'exercise_summary', db_path)
    new_ids = df['activity_id'][~df['activity_id'].isin(old_ids)]

    df_new = df[df['activity_id'].isin(new_ids)]

    if df_new.shape[0] > 0:
        with sqlite3.connect(db_path) as conn:
            df_new.to_sql(
                name='exercise_summary',
                con=conn,
                if_exists='append',
                index=False
            )

    print(f"Rejected existing ids: {old_ids}")

    return new_ids, old_ids


def create_exercise_table(db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS exercise(
                  activity_id   INT
                , time        INT
                , latitude    REAL
                , longitude   REAL
                , elevation   REAL
                , distance    REAL
                , velocity_smooth REAL
            );
        """)


def append_exercises(df, db_path=db_path):
    proposed_ids = list(set([int(x) for x in df['activity_id'].to_numpy()]))
    old_ids = existing_activity_ids(proposed_ids, 'exercise', db_path)
    new_ids = df['activity_id'][~df['activity_id'].isin(old_ids)]

    df_new = df[df['activity_id'].isin(new_ids)]

    if df_new.shape[0] > 0:
        with sqlite3.connect(db_path) as conn:
            df_new.to_sql(
                name='exercise',
                con=conn,
                if_exists='append',
                index=False
            )


def create_db(db_path=db_path):
    create_exercise_summary_table(db_path)
    create_exercise_table(db_path)
