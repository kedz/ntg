from .util import get_data_dir
import os
import urllib.request
import zipfile
import sys
import torch


def copy_from_pretrained(name, size, dest, vocab, verbose=False):

    if name.startswith("glove"):
        emb_iter = get_glove_iter(name, size)
    else:
        raise Exception("Invalid name argument.")

    replace_count = 0
    for word, lazy_embedding in emb_iter:
        idx = vocab[word] 
        if idx != vocab.unknown_index:
            dest[idx].copy_(lazy_embedding())
            replace_count += 1
    if verbose:
        print("Replaced {} of {} words ({:0.3f}%).".format(
            replace_count, vocab.size, replace_count / vocab.size * 100))

def get_glove_iter(name, size):
    path = get_glove_data_path(name, size)
    with zipfile.ZipFile(path) as zfp:

        if not "glove.840B.300d.txt" in zfp.namelist():
            raise Exception(
                "Bad file: {}\nTry deleting and run again.".format(path))
        with zfp.open("glove.840B.300d.txt") as fp:
            for line in fp:
                items = line.split()
                word = items[0].decode("utf8")
                def lazy_embedding():
                    return torch.FloatTensor([float(x) for x in items[1:]])
                yield word, lazy_embedding


def get_glove_data_path(name, size):
    if name == "glove.840B" and size == 300:
        filename = "{}.{}d.zip".format(name, size)
        path = os.path.join(get_data_dir(), "glove", filename)
    else:
        raise Exception()
    
    if not os.path.exists(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        url = "http://nlp.stanford.edu/data/{}".format(filename)

        response = urllib.request.urlopen(url)
        size = int(response.headers['content-length'])
        read = 0

        with open(path, "wb") as fp:
            while read < size:
                chunk = response.read(2048)
                read += len(chunk)
                fp.write(chunk)
                sys.stdout.write("\r{:0.3f}%".format(read / size * 100))
                sys.stdout.flush()
        print("")
    return path
        
#
#path = "glove.840B.300d.zip"

#
#
