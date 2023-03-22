from pathlib import Path
from configparser import ConfigParser

# config file should be in parent to src folder
config_path = Path(__file__).resolve().parents[1]/"config.ini"

def get_strava_config():
    config = ConfigParser()
    config.read(config_path)
    cfg = {}

    cfg['client_id'] = int(config['strava']['client_id'])
    cfg['client_secret'] = config['strava']['client_secret']
    cfg['access_token_path'] = config['strava']['access_token_path']
    cfg['days_before_limit'] = int(config['strava']['days_before_limit'])

    return cfg


def get_db_config():
    config = ConfigParser()
    config.read(config_path)
    dbconf = config['database']

    cfg = {}
    cfg['db_path'] =  dbconf['db_path']

    return cfg

