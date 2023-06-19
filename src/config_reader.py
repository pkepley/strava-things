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


def get_start_loc():
    config = ConfigParser()
    config.read(config_path)
    loc_conf = config['location']

    cfg = {}
    cfg['start_lon'] =  float(loc_conf['start_lon'])
    cfg['start_lat'] =  float(loc_conf['start_lat'])

    return cfg


def get_usual_route_file():
    config = ConfigParser()
    config.read(config_path)

    if config.has_section('route') and \
       config.has_option('route', 'route_file'):
        return Path(config['route']['route_file'])

    return None


def get_curve_clusterer_path():
    config = ConfigParser()
    config.read(config_path)

    if config.has_section('clusterer') and \
       config.has_option('clusterer', 'clusterer_path'):
        return Path(config['clusterer']['clusterer_path'])

    return None
