import os

def get_data_dir():
    '''
    Create data directory for toy datasets. Default location is ~/nt_data.
    To specify an alternative location set the NT_DATA environment variable.
    '''

    home_dir = os.path.expanduser("~")
    default_path = os.path.join(home_dir, "nt_data")
    data_dir = os.getenv('NT_DATA', default_path)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir

