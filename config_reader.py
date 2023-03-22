from configparser import ConfigParser

def get_strava_config():
    config = ConfigParser()
    config.read("config.ini")
    cfg = {}

    cfg['client_id'] = int(config['strava']['client_id'])
    cfg['client_secret'] = config['strava']['client_secret']
    cfg['access_token_path'] = config['strava']['access_token_path']

    return cfg


def get_db_config():
    config = ConfigParser()
    config.read("config.ini")
    dbconf = config['database']

    cfg = {}
    cfg['db_path'] =  dbconf['db_path']

    return cfg
