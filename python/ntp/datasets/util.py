import sys
import os
import urllib.request
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
