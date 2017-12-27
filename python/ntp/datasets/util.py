import sys
import os
import urllib.request
import requests
import io


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

def download_url_to_buffer(url, chunk_size=5096):
    response = urllib.request.urlopen(url)
    size = int(response.headers['content-length'])
    read = 0
    buffer = io.BytesIO()
    while read < size:
        chunk = response.read(chunk_size)
        read += len(chunk)
        buffer.write(chunk)
        sys.stdout.write("\rread {}/{} bytes ({:0.3f}%)".format(
            read, size, read / size * 100))
        sys.stdout.flush()
    print("")
    buffer.seek(0)
    return buffer

def download_url_to_file(url, path, chunk_size=5096):
    response = urllib.request.urlopen(url)
    size = int(response.headers['content-length'])
    read = 0
    with open(path, "wb") as fp:
        while read < size:
            chunk = response.read(chunk_size)
            read += len(chunk)
            fp.write(chunk)
            sys.stdout.write("\rread {}/{} bytes ({:0.3f}%)".format(
                read, size, read / size * 100))
            sys.stdout.flush()
        print("")

def download_google_drive_to_buffer(id, chunk_size=32768):

    URL = "https://docs.google.com/uc?export=download&fields=*"
    session = requests.Session()
    response = session.get(URL, params = {'id': id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(
                URL, params = {'id': id, 'confirm': value}, stream=True)

    read = 0
    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size):
        if chunk:
            buffer.write(chunk)
            read += len(chunk)
            sys.stdout.write("\rread {} bytes".format(
                read))
            sys.stdout.flush()
    print("")
    buffer.seek(0)
    return buffer
