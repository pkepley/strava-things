import time
from datetime import datetime, timedelta
import pickle
import pandas as pd
from stravalib.client import Client
import build_db as bdb
from config_reader import get_strava_config, get_db_config


# TODO: this script probably should be converted into
#       a few functions, possibly bundled into a class

# get the strava configuration
cfg = get_strava_config()
client_id = cfg['client_id']
client_secret = cfg['client_secret']
access_token_path = cfg['access_token_path']
days_before_limit = cfg['days_before_limit']

# get the db configuration
cfg = get_db_config()
db_path = cfg['db_path']

# set up the client
client = Client()

with open(access_token_path, 'rb') as f:
    access_token = pickle.load(f)

if time.time() > access_token['expires_at']:
    print('Token has expired, refreshing')
    refresh_response = client.refresh_access_token(
        client_id=client_id,
        client_secret=client_secret,
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
after_limit = datetime.now() - timedelta(days=days_before_limit)
after_limit = after_limit.strftime("%Y-%m-%d")
print(f"Pulling activities after {after_limit}")

activities = client.get_activities(after = after_limit)
activities = [activity.to_dict() for activity in activities]

activity_cols = [
    'name', 'id', 'distance', 'moving_time', 'elapsed_time',
    'total_elevation_gain', 'elev_high', 'elev_low', 'type', 'start_date',
    'timezone', 'start_latlng', 'end_latlng'
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
new_activity_ids, _ = bdb.append_exercise_summary(df_activities, db_path=db_path)

# get the *last* activity
stream_columns = ['time', 'distance', 'latlng', 'altitude', 'velocity_smooth']

for activity_id in new_activity_ids:
    activity_id_in_db = bdb.existing_activity_ids([activity_id], "exercise", db_path=db_path)

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

    bdb.append_exercises(df_activity, db_path=db_path)
