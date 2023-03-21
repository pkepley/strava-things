import time
import pickle
import requests
import pandas as pd
import sqlite3
from stravalib.client import Client
#import build_db as bd
import build_db as bdb
import config


# TODO: should be configurable
access_token_path = 'access_token.pickle'

#
client = Client()

# TODO: should
with open(access_token_path, 'rb') as f:
    access_token = pickle.load(f)

if time.time() > access_token['expires_at']:
    print('Token has expired, refreshing')
    refresh_response = client.refresh_access_token(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        refresh_token=access_token['refresh_token']
    )
    access_token = refresh_response

    with open(access_token_path, 'wb') as f:
        pickle.dump(refresh_response, f)

    print('Refreshed token saved to file')
    client.access_token = refresh_response['access_token']
    client.refresh_token = refresh_response['refresh_token']
    client.token_expires_at = refresh_response['expires_at']

else:
    print('Token still valid, expires at {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S %Z",
                      time.localtime(access_token['expires_at']))
    ))
    client.access_token = access_token['access_token']
    client.refresh_token = access_token['refresh_token']
    client.token_expires_at = access_token['expires_at']

# get activities
activities = client.get_activities(after = '2023-01-01')
activities = [activity.to_dict() for activity in activities]

activity_cols = [
    'name', 'id', 'distance', 'moving_time', 'elapsed_time',
    'total_elevation_gain', 'elev_high', 'elev_low',
    'type',
    'start_date',
    'timezone',
    'start_latlng',
    'end_latlng'
]

df_activities = pd.DataFrame.from_dict({
    k: [a[k] for a in activities] for k in activity_cols
})
df_activities[['start_latitude', 'start_longitude']] = (
    df_activities['start_latlng']
    .apply(lambda s: [pd.NA, pd.NA] if s is None else [float(x) for x in s.split(",")])
    .to_list()
)
df_activities[['end_latitude', 'end_longitude']] = (
    df_activities['end_latlng']
    .apply(lambda s: [pd.NA, pd.NA] if s is None else [float(x) for x in s.split(",")])
    .to_list()
)
df_activities.drop(columns=['start_latlng', 'end_latlng'], inplace=True)
df_activities['start_date_utc'] = pd.to_datetime(df_activities['start_date'])
df_activities['start_datetime'] = df_activities.apply(lambda row: row['start_date_utc'].tz_convert(row['timezone']), axis=1)
df_activities['date'] = df_activities['start_datetime'].dt.strftime('%Y-%m-%d')
df_activities.drop(columns=['start_datetime', 'start_date'], inplace=True)
df_activities.rename(columns={'start_date_utc': 'start_datetime_utc', 'id': 'activity_id'}, inplace=True)

bdb.create_db()
new_activity_ids, _ = bdb.append_exercise_summary(df_activities)

# get the *last* activity
stream_columns = ['time', 'distance', 'latlng', 'altitude', 'velocity_smooth']

for activity_id in new_activity_ids:
    activity_id_in_db = bdb.existing_activity_ids([activity_id], "exercise", db_path="my_strava.db")

    if activity_id_in_db:
        print(f"Activity {activity_id} already in db! Skipping")
        continue

    print(f"Pulling activity_id {activity_id}")
    streams = client.get_activity_streams(
        activity_id,
        types=stream_columns,
        resolution='high'
    )

    if streams is None:
        print(f"Something is fishy with {activity_id}")
        continue

    df_activity = pd.DataFrame({
        k: streams[k].data for k in stream_columns
        if k in streams.keys()
    })
    df_activity[['latitude', 'longitude']] = df_activity['latlng'].to_list()
    df_activity.drop(columns=['latlng'], inplace=True)
    df_activity['id'] = activity_id
    df_activity.rename(columns={'altitude': 'elevation', 'id': 'activity_id'}, inplace=True)

    bdb.append_exercises(df_activity)
